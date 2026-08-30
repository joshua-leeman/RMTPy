import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.special import xlogy

from rmtpy.ensembles import GOE, GUE
from rmtpy.simulations.cdo_evolution import CDOEvolutionSimulation
from rmtpy.simulations.cdo_evolution.cdo_dynamics import (
    CDODynamicsData,
    CDOInformationPlot,
    CDOProbabilitiesPlot,
    CDOPuritiesPlot,
)
from rmtpy.simulations.spectral_statistics.spectral_form_factors import (
    FormFactorsData,
    FormFactorsPlot,
)


def compute_direct_cdo_dynamics(
    *,
    ensemble,
    realizs: int,
    times: np.ndarray,
    initial_state: np.ndarray,
) -> tuple[np.ndarray, ...]:
    states = []
    for eigenvalues, eigenvectors in ensemble.eigsys_stream(realizs):
        rotated_state = eigenvectors.conj().T @ initial_state
        states.append(
            (np.exp(-1j * np.outer(times, eigenvalues)) * rotated_state) @ eigenvectors.T
        )

    states_array = np.asarray(states)
    state_probabilities = np.abs(states_array) ** 2
    state_probabilities /= np.sum(state_probabilities, axis=2, keepdims=True)
    density_operators = (
        np.einsum(
            "rti,rtj->tij",
            states_array,
            states_array.conj(),
            optimize=True,
        )
        / realizs
    )
    density_operators /= np.trace(
        density_operators,
        axis1=1,
        axis2=2,
    ).real[:, np.newaxis, np.newaxis]

    probabilities = np.diagonal(
        density_operators,
        axis1=1,
        axis2=2,
    ).real.copy()
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    classical_purity = np.sum(probabilities**2, axis=1)

    eigenvalues = np.linalg.eigvalsh(density_operators)
    eigenvalues[eigenvalues < 0.0] = 0.0
    eigenvalues /= np.sum(eigenvalues, axis=1, keepdims=True)
    quantum_purity = np.sum(eigenvalues**2, axis=1)
    entropy = -np.sum(xlogy(eigenvalues, eigenvalues), axis=1)
    kl_divergence = np.mean(
        np.sum(
            xlogy(probabilities[np.newaxis, ...], probabilities[np.newaxis, ...])
            - xlogy(probabilities[np.newaxis, ...], state_probabilities),
            axis=2,
        ),
        axis=0,
    )
    return (
        probabilities,
        classical_purity,
        quantum_purity,
        entropy,
        kl_divergence,
    )


class CDOEvolutionTests(unittest.TestCase):
    def test_dynamics_match_direct_calculation(self) -> None:
        simulation = CDOEvolutionSimulation(
            ensemble=GUE(num_majoranas=4, seed=4321),
            realizs=3,
            num_times=7,
            time_chunk_size=2,
        )
        reference_ensemble = GUE(num_majoranas=4, seed=4321)
        dynamics = simulation.outputs.dynamics.data
        original_times = dynamics.times.copy()

        simulation.realize_monte_carlo_simulation()
        simulation.calculate_statistics()
        expected = compute_direct_cdo_dynamics(
            ensemble=reference_ensemble,
            realizs=simulation.realizs,
            times=dynamics.times,
            initial_state=simulation.initial_state,
        )

        np.testing.assert_array_equal(dynamics.times, original_times)
        for actual, reference in zip(
            (
                dynamics.probabilities,
                dynamics.classical_purity,
                dynamics.quantum_purity,
                dynamics.entropy,
                dynamics.kl_divergence,
            ),
            expected,
            strict=True,
        ):
            np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-12)

        self.assertEqual(dynamics.realizs, simulation.realizs)
        np.testing.assert_allclose(
            np.sum(dynamics.probabilities, axis=1),
            1.0,
        )
        self.assertTrue(np.all(dynamics.entropy >= 0.0))
        self.assertTrue(np.all(dynamics.kl_divergence >= 0.0))

        state_factor_simulation = CDOEvolutionSimulation(
            ensemble=GUE(num_majoranas=6, seed=8765),
            realizs=2,
            num_times=5,
            time_chunk_size=2,
        )
        state_factor_simulation.realize_monte_carlo_simulation()
        state_factor_simulation.calculate_statistics()
        state_factor_dynamics = state_factor_simulation.outputs.dynamics.data
        state_factor_expected = compute_direct_cdo_dynamics(
            ensemble=GUE(num_majoranas=6, seed=8765),
            realizs=state_factor_simulation.realizs,
            times=state_factor_dynamics.times,
            initial_state=state_factor_simulation.initial_state,
        )
        self.assertEqual(state_factor_simulation.outputs.accumulation_mode, "states")
        for actual, reference in zip(
            (
                state_factor_dynamics.probabilities,
                state_factor_dynamics.classical_purity,
                state_factor_dynamics.quantum_purity,
                state_factor_dynamics.entropy,
                state_factor_dynamics.kl_divergence,
            ),
            state_factor_expected,
            strict=True,
        ):
            np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-12)

    def test_time_zero_and_rank_one_are_numerically_stable(self) -> None:
        simulation = CDOEvolutionSimulation(
            ensemble=GOE(num_majoranas=4, seed=123),
            realizs=1,
            num_times=5,
            time_chunk_size=2,
        )
        simulation.realize_monte_carlo_simulation()
        simulation.calculate_statistics()
        dynamics = simulation.outputs.dynamics.data

        np.testing.assert_allclose(
            dynamics.probabilities[0],
            np.array([1.0, 0.0]),
            atol=1e-15,
        )
        np.testing.assert_allclose(dynamics.classical_purity[0], 1.0)
        np.testing.assert_allclose(dynamics.quantum_purity, 1.0)
        np.testing.assert_allclose(dynamics.entropy, 0.0, atol=1e-14)
        np.testing.assert_allclose(
            dynamics.kl_divergence,
            0.0,
            atol=1e-14,
        )
        self.assertTrue(
            all(
                np.all(np.isfinite(value))
                for value in (
                    dynamics.probabilities,
                    dynamics.classical_purity,
                    dynamics.quantum_purity,
                    dynamics.entropy,
                    dynamics.kl_divergence,
                )
            )
        )

    def test_reverse_kl_is_infinite_for_missing_realization_support(self) -> None:
        simulation = CDOEvolutionSimulation(
            ensemble=GOE(num_majoranas=4),
            realizs=2,
            num_times=1,
        )
        simulation.outputs.reset()
        simulation.outputs.add_evolved_states(np.array([[1.0, 0.0]]))
        simulation.outputs.add_evolved_states(np.array([[0.0, 1.0]]))
        simulation.outputs.calculate_dynamics()
        dynamics = simulation.outputs.dynamics.data

        np.testing.assert_allclose(
            dynamics.probabilities[0],
            np.array([0.5, 0.5]),
        )
        self.assertTrue(np.isposinf(dynamics.kl_divergence[0]))

    def test_outputs_are_native_and_memory_adaptive(self) -> None:
        state_accumulator = CDOEvolutionSimulation(
            ensemble=GOE(num_majoranas=4),
            realizs=1,
            num_times=3,
        )
        density_accumulator = CDOEvolutionSimulation(
            ensemble=GOE(num_majoranas=4),
            realizs=2,
            num_times=3,
        )
        retained_states = CDOEvolutionSimulation(
            ensemble=GOE(num_majoranas=4, seed=17),
            realizs=2,
            num_times=3,
            retain_evolved_states=True,
        )

        self.assertEqual(state_accumulator.outputs.accumulation_mode, "states")
        self.assertEqual(
            density_accumulator.outputs.accumulation_mode,
            "density_operator",
        )
        self.assertEqual(
            tuple(state_accumulator.iter_observables()),
            (state_accumulator.outputs.dynamics,),
        )
        self.assertIs(
            state_accumulator.get_data("cdo_dynamics"),
            state_accumulator.outputs.dynamics.data,
        )

        self.assertEqual(retained_states.outputs.accumulation_mode, "states")
        self.assertEqual(len(tuple(retained_states.iter_observables())), 2)
        retained_observable = retained_states.outputs.evolved_states
        self.assertIsNotNone(retained_observable)
        retained_states.realize_monte_carlo_simulation()
        retained_states.calculate_statistics()
        if retained_observable is None:
            self.fail("Expected the evolved-states observable to be configured.")
        np.testing.assert_allclose(
            np.sum(np.abs(retained_observable.data.states) ** 2, axis=2),
            1.0,
        )

    def test_initial_state_is_validated_copied_and_read_only(self) -> None:
        source = np.array([0.0, 1.0])
        simulation = CDOEvolutionSimulation(
            ensemble=GOE(num_majoranas=4),
            realizs=1,
            initial_state=source,
            num_times=2,
        )
        source[1] = 0.0

        np.testing.assert_array_equal(simulation.initial_state, np.array([0.0, 1.0]))
        self.assertFalse(np.shares_memory(source, simulation.initial_state))
        self.assertFalse(simulation.initial_state.flags.writeable)
        with self.assertRaises(ValueError):
            simulation.initial_state[0] = 1.0

        for invalid_state in (
            [1.0],
            [1.0, np.nan],
            [1.0, 1.0],
            [0.0, 0.0],
        ):
            with self.subTest(initial_state=invalid_state), self.assertRaises(ValueError):
                CDOEvolutionSimulation(
                    ensemble=GOE(num_majoranas=4),
                    realizs=1,
                    initial_state=invalid_state,
                    num_times=2,
                )

    def test_result_round_trip(self) -> None:
        simulation = CDOEvolutionSimulation(
            ensemble=GOE(num_majoranas=4, seed=99),
            realizs=2,
            num_times=4,
        )
        simulation.realize_monte_carlo_simulation()
        simulation.calculate_statistics()

        with tempfile.TemporaryDirectory() as tmp_dir:
            simulation.save_data(tmp_dir)
            path = Path(tmp_dir) / "cdo_dynamics" / "cdo_dynamics_data.npz"
            restored = CDODynamicsData.load(path)

        dynamics = simulation.outputs.dynamics.data
        np.testing.assert_array_equal(restored.times, dynamics.times)
        np.testing.assert_allclose(
            restored.probabilities,
            dynamics.probabilities,
        )
        np.testing.assert_allclose(
            restored.kl_divergence,
            dynamics.kl_divergence,
        )
        self.assertEqual(restored.realizs, simulation.realizs)

    def test_plot_views_share_the_raw_form_factor_time_axis(self) -> None:
        simulation = CDOEvolutionSimulation(
            ensemble=GOE(num_majoranas=4, seed=2468),
            realizs=2,
            num_times=7,
            time_chunk_size=2,
        )
        simulation.realize_monte_carlo_simulation()
        simulation.calculate_statistics()

        observable = simulation.outputs.dynamics
        self.assertEqual(
            observable.plot_classes,
            (CDOProbabilitiesPlot, CDOPuritiesPlot, CDOInformationPlot),
        )

        runtime_args = {"ensemble": simulation.ensemble}
        form_factor_plot = FormFactorsPlot(
            data=FormFactorsData(
                dimension=simulation.ensemble.dimension,
                logD_time_support=(-0.5, 1.5),
                scale=simulation.time_scale,
                num_times=7,
            ),
            runtime_simulation_args=runtime_args.copy(),
        )
        form_factor_plot.set_derived_attributes()

        for plot_cls in observable.plot_classes:
            with self.subTest(plot_cls=plot_cls):
                plot = plot_cls(
                    data=observable.data,
                    runtime_simulation_args=runtime_args.copy(),
                )
                self.assertEqual(plot.axes.xlabel, form_factor_plot.axes.xlabel)
                self.assertEqual(
                    plot.axes.xtick_labels,
                    form_factor_plot.axes.xtick_labels,
                )

                with patch.object(plot, "finish_plot"):
                    plot.plot(path="unused")

                np.testing.assert_allclose(plot.xlim, form_factor_plot.xlim)
                np.testing.assert_allclose(
                    plot.axes.xticks,
                    form_factor_plot.axes.xticks,
                )
                self.assertEqual(plot.ax.get_xscale(), "log")
                np.testing.assert_array_equal(
                    plot.ax.lines[0].get_xdata(),
                    simulation.outputs.dynamics.data.times[1:],
                )

        probability_plot = CDOProbabilitiesPlot(
            data=observable.data,
            runtime_simulation_args=runtime_args.copy(),
        )
        purity_plot = CDOPuritiesPlot(
            data=observable.data,
            runtime_simulation_args=runtime_args.copy(),
        )
        information_plot = CDOInformationPlot(
            data=observable.data,
            runtime_simulation_args=runtime_args.copy(),
        )
        for plot in (probability_plot, purity_plot, information_plot):
            with patch.object(plot, "finish_plot"):
                plot.plot(path="unused")

        self.assertEqual(probability_plot.ax.get_yscale(), "log")
        self.assertEqual(purity_plot.ax.get_yscale(), "log")
        self.assertEqual(information_plot.ax.get_yscale(), "linear")
        self.assertEqual(
            len(probability_plot.ax.lines),
            simulation.ensemble.dimension + 1,
        )
        self.assertEqual(len(purity_plot.ax.lines), 2)
        self.assertEqual(len(information_plot.ax.lines), 2)
        self.assertEqual(probability_plot.file_name, "cdo_probabilities_plot")
        self.assertEqual(purity_plot.file_name, "cdo_purities_plot")
        self.assertEqual(information_plot.file_name, "cdo_information_plot")

    def test_observable_saves_all_three_transient_plot_views(self) -> None:
        simulation = CDOEvolutionSimulation(
            ensemble=GOE(num_majoranas=4),
            realizs=1,
            num_times=3,
        )
        observable = simulation.outputs.dynamics

        with (
            patch.object(CDOProbabilitiesPlot, "plot", autospec=True) as probabilities,
            patch.object(CDOPuritiesPlot, "plot", autospec=True) as purities,
            patch.object(CDOInformationPlot, "plot", autospec=True) as information,
        ):
            observable.save_plot(
                Path("unused"),
                simulation_args={"ensemble": simulation.ensemble},
            )

        probabilities.assert_called_once()
        purities.assert_called_once()
        information.assert_called_once()
        self.assertEqual(
            probabilities.call_args.kwargs["path"],
            Path("unused") / "cdo_dynamics",
        )


if __name__ == "__main__":
    unittest.main()
