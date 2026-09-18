# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportUnusedCallResult=false, reportPrivateUsage=false, reportImplicitStringConcatenation=false, reportMissingParameterType=false, reportUnnecessaryIsInstance=false, reportImplicitOverride=false, reportExplicitAny=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false

import tempfile
import unittest
from copy import deepcopy
from unittest.mock import patch

import numpy as np

from rmtpy.compounds import CompoundEnsemble
from rmtpy.ensembles import GOE
from rmtpy.simulations.base_simulation import SimulationExecutionState as ExecutionState
from rmtpy.simulations.transmission_coefficients_simulation import (
    TransmissionCoefficientsResult,
    TransmissionCoefficientsSimulation,
    load_transmission_coefficients_result,
    plot_transmission_coefficients_result,
    save_transmission_coefficients_result,
)
from rmtpy.simulations.transmission_coefficients_simulation.transmission_coefficients import (
    TransmissionCoefficientsPlot,
)
from rmtpy.simulations.transmission_coefficients_simulation.weisskopf_estimate import (
    WeisskopfEstimatePlot,
)


def build_compound(*, seed: int = 123) -> CompoundEnsemble:
    return CompoundEnsemble(
        ensemble=GOE(num_majoranas=4, seed=seed),
        coupling_strengths=np.array([0.75, 1.25]),
    )


def scattering_matrix(
    energies: np.ndarray,
    diagonal: tuple[complex, complex],
) -> np.ndarray:
    matrices = np.zeros((len(energies), 2, 2), dtype=np.complex128)
    matrices[:, 0, 0] = diagonal[0]
    matrices[:, 1, 1] = diagonal[1]
    return matrices


class TransmissionCoefficientsTests(unittest.TestCase):
    def test_complex_average_precedes_modulus_and_channel_routing(self) -> None:
        simulation = TransmissionCoefficientsSimulation(
            compound=build_compound(),
            realizs=2,
            channel_indices=(1, 0),
        )
        first = scattering_matrix(simulation.energies, (1.0, 0.5 + 0.5j))
        second = scattering_matrix(simulation.energies, (-1.0, -0.5 + 0.5j))

        with patch.object(
            CompoundEnsemble,
            "scattering_matrix_stream",
            return_value=iter(((first, np.empty(0)), (second, np.empty(0)))),
        ):
            result = simulation.execute()

        self.assertIsInstance(result, TransmissionCoefficientsResult)
        self.assertIs(result.energies, simulation.energies)
        self.assertEqual(
            tuple(channel.channel_index for channel in result.by_channel),
            (1, 0),
        )
        np.testing.assert_allclose(
            result.by_channel[0].data.average_scattering_diagonal,
            0.5j,
        )
        np.testing.assert_allclose(
            result.by_channel[0].data.transmission_coefficients,
            0.75,
        )
        np.testing.assert_allclose(
            result.by_channel[1].data.average_scattering_diagonal,
            0.0,
        )
        np.testing.assert_allclose(
            result.by_channel[1].data.transmission_coefficients,
            1.0,
        )
        np.testing.assert_allclose(
            result.weisskopf_estimate.transmission_coefficients,
            np.tile((1.0, 0.75), (len(simulation.energies), 1)),
        )
        self.assertEqual(
            tuple(data.file_name for data in result.iterate_data()),
            (
                "transmission_coefficients_data",
                "transmission_coefficients_data",
                "weisskopf_estimate_data",
            ),
        )

    def test_weisskopf_uses_all_channels_and_leading_weight_density(self) -> None:
        simulation = TransmissionCoefficientsSimulation(
            compound=build_compound(),
            realizs=1,
            channel_indices=(0,),
        )
        matrices = scattering_matrix(simulation.energies, (0.5, 0.0))
        with patch.object(
            CompoundEnsemble,
            "scattering_matrix_stream",
            return_value=iter(((matrices, np.empty(0)),)),
        ):
            result = simulation.execute()

        weisskopf = result.weisskopf_estimate
        self.assertEqual(len(result.by_channel), 1)
        self.assertEqual(weisskopf.num_channels, 2)
        np.testing.assert_allclose(
            weisskopf.transmission_coefficients,
            np.tile((0.75, 1.0), (len(simulation.energies), 1)),
        )
        density = simulation.compound.ensemble.spectral_density.weight_pdf(
            simulation.energies
        )
        valid = np.isfinite(density) & (density > 0.0)
        expected = 1.75 / (2.0 * simulation.compound.ensemble.dimension * density[valid])
        np.testing.assert_allclose(weisskopf.weisskopf_estimate[valid], expected)
        self.assertTrue(np.all(np.isnan(weisskopf.weisskopf_estimate[~valid])))

    def test_invalid_density_values_remain_undefined(self) -> None:
        simulation = TransmissionCoefficientsSimulation(
            compound=build_compound(),
            realizs=1,
        )
        density = np.ones(len(simulation.energies))
        density[:3] = (0.0, -1.0, np.nan)
        matrices = scattering_matrix(simulation.energies, (0.0, 0.0))

        with (
            patch.object(
                type(simulation.compound.ensemble.spectral_density),
                "weight_pdf",
                return_value=density,
            ),
            patch.object(
                CompoundEnsemble,
                "scattering_matrix_stream",
                return_value=iter(((matrices, np.empty(0)),)),
            ),
        ):
            result = simulation.execute()

        self.assertTrue(
            np.all(np.isnan(result.weisskopf_estimate.mean_level_spacings[:3]))
        )
        self.assertTrue(
            np.all(np.isnan(result.weisskopf_estimate.weisskopf_estimate[:3]))
        )

    def test_exact_energy_grid_channel_validation_and_selective_allocation(
        self,
    ) -> None:
        simulation = TransmissionCoefficientsSimulation(
            compound=build_compound(),
            realizs=1,
            channel_indices=(1,),
        )
        expected = np.linspace(
            *simulation.compound.ensemble.spectral_density.plot_range,
            100,
            dtype=np.float64,
        )
        np.testing.assert_array_equal(simulation.energies, expected)
        self.assertFalse(simulation.energies.flags.writeable)
        self.assertFalse(hasattr(simulation, "outputs"))

        matrices = scattering_matrix(simulation.energies, (0.0, 0.25))
        with patch.object(
            CompoundEnsemble,
            "scattering_matrix_stream",
            return_value=iter(((matrices, np.empty(0)),)),
        ):
            result = simulation.execute()
        self.assertEqual(
            tuple(channel.channel_index for channel in result.by_channel), (1,)
        )

        compound = build_compound()
        for channel_indices in ((), (0, 0), (-1,), (compound.num_channels,), ((0, 1),)):
            with (
                self.subTest(channel_indices=channel_indices),
                self.assertRaises(ValueError),
            ):
                TransmissionCoefficientsSimulation(
                    compound=compound,
                    realizs=1,
                    channel_indices=channel_indices,
                )

    def test_malformed_stream_is_a_terminal_failure(self) -> None:
        simulation = TransmissionCoefficientsSimulation(
            compound=build_compound(),
            realizs=1,
        )
        malformed = np.zeros((len(simulation.energies), 1, 1), dtype=np.complex128)
        with (
            patch.object(
                CompoundEnsemble,
                "scattering_matrix_stream",
                return_value=iter(((malformed, np.empty(0)),)),
            ),
            self.assertRaisesRegex(ValueError, "must have shape"),
        ):
            simulation.execute()

        self.assertEqual(simulation.execution_state, ExecutionState.FAILED)
        with self.assertRaisesRegex(RuntimeError, "only once"):
            simulation.execute()

    def test_seeded_stream_and_rng_match_direct_scattering(self) -> None:
        simulation = TransmissionCoefficientsSimulation(
            compound=build_compound(seed=314159),
            realizs=2,
            channel_indices=(0,),
        )
        control = build_compound(seed=314159)
        initial_rng_state = deepcopy(simulation.compound.rng_state)
        diagonal_sum = np.zeros(
            (len(simulation.energies), control.num_channels),
            dtype=np.complex128,
        )
        for matrices, _ in control.scattering_matrix_stream(
            realizs=2,
            energies=simulation.energies,
        ):
            diagonal_sum += np.diagonal(matrices, axis1=1, axis2=2)

        result = simulation.execute()
        expected_average = diagonal_sum / simulation.realizs
        expected_transmission = 1.0 - np.abs(expected_average) ** 2
        np.clip(expected_transmission, 0.0, 1.0, out=expected_transmission)

        self.assertEqual(simulation.compound.rng_state, control.rng_state)
        self.assertEqual(result.context.rng["initial_state"], initial_rng_state)
        self.assertEqual(result.context.rng["final_state"], control.rng_state)
        np.testing.assert_allclose(
            result.weisskopf_estimate.average_scattering_diagonal,
            expected_average,
        )
        np.testing.assert_allclose(
            result.weisskopf_estimate.transmission_coefficients,
            expected_transmission,
        )

    def test_persistence_plotting_and_run_consumers(self) -> None:
        simulation = TransmissionCoefficientsSimulation(
            compound=build_compound(seed=90),
            realizs=1,
            channel_indices=(0,),
        )
        result = simulation.execute()
        completed_rng_state = deepcopy(simulation.compound.rng_state)

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = save_transmission_coefficients_result(result, out_dir=tmp_dir)
            restored = load_transmission_coefficients_result(run_dir)
            np.testing.assert_allclose(
                restored.by_channel[0].data.transmission_coefficients,
                result.by_channel[0].data.transmission_coefficients,
            )

            with (
                patch.object(
                    TransmissionCoefficientsPlot, "plot", autospec=True
                ) as channel_plot,
                patch.object(
                    WeisskopfEstimatePlot, "plot", autospec=True
                ) as weisskopf_plot,
            ):
                plot_transmission_coefficients_result(
                    result,
                    out_dir=tmp_dir,
                    channel_indices=0,
                    include_weisskopf=False,
                )
            channel_plot.assert_called_once()
            weisskopf_plot.assert_not_called()
        self.assertEqual(simulation.compound.rng_state, completed_rng_state)

        data_only = TransmissionCoefficientsSimulation(
            compound=build_compound(seed=91),
            realizs=1,
        )
        with patch("rmtpy.simulations.persistence.save_run") as save_result:
            run_result = data_only.execute()
        self.assertIs(run_result, data_only.result)
        save_result.assert_not_called()


if __name__ == "__main__":
    unittest.main()
