# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportUnusedCallResult=false, reportPrivateUsage=false, reportImplicitStringConcatenation=false, reportMissingParameterType=false, reportUnnecessaryIsInstance=false, reportImplicitOverride=false, reportExplicitAny=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false

import tempfile
import unittest
from copy import deepcopy
from unittest.mock import patch

import numpy as np

from rmtpy.compounds import Compound
from rmtpy.ensembles import GOE
from rmtpy.simulations.base_simulation import SimulationExecutionState as ExecutionState
from rmtpy.simulations.time_delay_statistics import (
    TimeDelayStatisticsRequest,
    TimeDelayStatisticsResult,
    TimeDelayStatisticsSimulation,
    load_time_delay_statistics_result,
    plot_time_delay_statistics_result,
    save_time_delay_statistics_result,
)
from rmtpy.simulations.time_delay_statistics.time_delay_histograms import (
    TimeDelayHistogramPlot,
    UnfoldedTimeDelayHistogramPlot,
)
from rmtpy.simulations.time_delay_statistics.time_delay_statistics_simulation import (
    unfold_delay_values,
)
from rmtpy.simulations.unfolding import TruncatedPolynomialCdfFactory


def build_compound(*, max_degree: int = 0, seed: int = 123) -> Compound:
    return Compound(
        ensemble=GOE(
            num_majoranas=4,
            max_spectral_polynomial_degree=max_degree,
            seed=seed,
        ),
        coupling_strengths=np.array([0.75, 1.25]),
    )


def histogram_counts(samples: list[np.ndarray], bins: np.ndarray) -> np.ndarray:
    counts = np.zeros(len(bins) - 1, dtype=np.int64)
    for sample in samples:
        indices = np.searchsorted(bins, sample, side="right") - 1
        valid = (indices >= 0) & (indices < len(counts))
        np.add.at(counts, indices[valid], 1)
    return counts


class TimeDelayStatisticsTests(unittest.TestCase):
    def test_energies_are_copied_read_only_unique_and_path_safe(self) -> None:
        source = np.array([-0.25, 0.125])
        simulation = TimeDelayStatisticsSimulation(
            compound=build_compound(),
            realizs=1,
            energies=source,
        )
        source[0] = -0.5
        np.testing.assert_array_equal(simulation.energies, np.array([-0.25, 0.125]))
        self.assertFalse(np.shares_memory(source, simulation.energies))
        self.assertFalse(simulation.energies.flags.writeable)
        with self.assertRaises(ValueError):
            simulation.energies[0] = 0.0

        close = TimeDelayStatisticsSimulation(
            compound=build_compound(),
            realizs=1,
            energies=(0.123456, 0.123457),
        )
        np.testing.assert_array_equal(close.energies, [0.123456, 0.123457])

        for energies in ((-0.0, 0.0), (0.1, 0.1)):
            with self.subTest(energies=energies), self.assertRaises(ValueError):
                TimeDelayStatisticsSimulation(
                    compound=build_compound(),
                    realizs=1,
                    energies=energies,
                )

    def test_reciprocal_width_unfolding_sequence_and_edge_filtering(self) -> None:
        delays = np.array([2.0, 4.0, np.nan, np.inf, 0.0, -1.0])
        unfolded = unfold_delay_values(
            delays,
            energy=0.25,
            cdf=lambda values: values,
            dimension=2,
        )
        np.testing.assert_allclose(unfolded, np.array([1.0, 2.0]))

        empty = unfold_delay_values(
            np.array([0.0, -1.0, np.nan]),
            energy=0.0,
            cdf=lambda values: values,
            dimension=2,
        )
        self.assertEqual(empty.size, 0)

    def test_degree_by_energy_axes_and_canonical_metadata(self) -> None:
        simulation = TimeDelayStatisticsSimulation(
            compound=build_compound(max_degree=2, seed=41),
            realizs=1,
            energies=(-0.25, 0.0, 0.125),
        )
        result = simulation.execute()

        self.assertIsInstance(result, TimeDelayStatisticsResult)
        self.assertIs(result.energies, simulation.energies)
        self.assertFalse(hasattr(simulation, "outputs"))
        self.assertEqual(simulation.truncated_degrees, (2,))
        self.assertEqual(len(result.raw), 3)
        self.assertEqual(len(result.weight), 3)
        self.assertEqual(len(result.average_by_degree), 1)
        self.assertEqual(len(result.variate_by_degree), 1)
        self.assertEqual(len(tuple(result.iterate_data())), 12)

        for mode, group in (("raw", result.raw), ("weight", result.weight)):
            for energy_index, energy_result in enumerate(group):
                self.assertEqual(energy_result.energy_index, energy_index)
                self.assertEqual(energy_result.energy, simulation.energies[energy_index])
                self.assertEqual(
                    energy_result.histogram.metadata["unfolding"],
                    mode,
                )
        for mode, degree_results in (
            ("average", result.average_by_degree),
            ("variate", result.variate_by_degree),
        ):
            self.assertEqual(degree_results[0].degree, 2)
            for energy_result in degree_results[0].by_energy:
                self.assertEqual(energy_result.histogram.metadata["unfolding"], mode)
                self.assertEqual(energy_result.histogram.metadata["degree"], 2)

        self.assertEqual(
            tuple(item.energy_index for item in result.raw),
            tuple(range(len(simulation.energies))),
        )

    def test_open_channel_shape_and_closed_eigenvalue_coefficients(self) -> None:
        simulation = TimeDelayStatisticsSimulation(
            compound=build_compound(max_degree=2),
            realizs=1,
            energies=(-0.1, 0.1),
            request=TimeDelayStatisticsRequest(
                unfolding_modes=("variate",),
                degrees=(2,),
            ),
        )
        delays = np.full((2, simulation.compound.num_channels), 1.5)
        closed_eigenvalues = np.array([-0.75, 0.5])
        density = simulation.compound.ensemble.spectral_density
        compute_coefficients = density.compute_variate_coeffs

        with (
            patch.object(
                Compound,
                "time_delays_stream",
                return_value=iter(((delays, closed_eigenvalues),)),
            ),
            patch.object(
                type(density),
                "compute_variate_coeffs",
                autospec=True,
                side_effect=lambda _density, values: compute_coefficients(values),
            ) as compute,
        ):
            result = simulation.execute()

        np.testing.assert_array_equal(compute.call_args.args[1], closed_eigenvalues)
        self.assertEqual(len(result.variate_by_degree[0].by_energy), 2)

        failed = TimeDelayStatisticsSimulation(
            compound=build_compound(),
            realizs=1,
            energies=(-0.1, 0.1),
        )
        malformed = np.ones((2, 1))
        with (
            patch.object(
                Compound,
                "time_delays_stream",
                return_value=iter(((malformed, closed_eigenvalues),)),
            ),
            self.assertRaisesRegex(ValueError, "must have shape"),
        ):
            failed.execute()
        self.assertEqual(failed.execution_state, ExecutionState.FAILED)

    def test_selective_raw_request_omits_unfolding_and_calibration(self) -> None:
        simulation = TimeDelayStatisticsSimulation(
            compound=build_compound(max_degree=2),
            realizs=1,
            energies=(0.0,),
            request=TimeDelayStatisticsRequest(unfolding_modes=("raw",)),
        )
        density = simulation.compound.ensemble.spectral_density
        delays = np.ones((1, simulation.compound.num_channels))
        with (
            patch.object(
                Compound,
                "time_delays_stream",
                return_value=iter(((delays, np.array([-0.5, 0.5])),)),
            ),
            patch.object(
                type(density),
                "compute_variate_coeffs",
                side_effect=AssertionError("coefficients were unrequested"),
            ),
            patch.object(
                type(density),
                "_compute_average_coeffs",
                side_effect=AssertionError("calibration was unrequested"),
            ),
            patch(
                "rmtpy.simulations.time_delay_statistics."
                "time_delay_statistics_simulation.unfold_delay_values",
                side_effect=AssertionError("unfolding was unrequested"),
            ),
        ):
            result = simulation.execute()

        self.assertEqual(len(result.raw), 1)
        self.assertEqual(result.weight, ())
        self.assertEqual(result.average_by_degree, ())
        self.assertEqual(result.variate_by_degree, ())
        self.assertEqual(len(tuple(result.iterate_data())), 1)

    def test_average_calibration_precedes_seeded_delay_stream(self) -> None:
        simulation = TimeDelayStatisticsSimulation(
            compound=build_compound(max_degree=2, seed=314159),
            realizs=2,
            energies=(-0.1, 0.1),
        )
        control = build_compound(max_degree=2, seed=314159)
        initial_rng_state = deepcopy(simulation.compound.rng_state)
        factory = TruncatedPolynomialCdfFactory(
            density=control.ensemble.spectral_density,
            degrees=(2,),
            density_name="spectral",
        )
        average_cdf = factory.average_interpolators()[0]
        samples: list[np.ndarray] = []
        variate_cdfs = []
        for delays, closed_eigenvalues in control.time_delays_stream(
            energies=simulation.energies,
            realizs=2,
        ):
            samples.append(delays.copy())
            coefficients = control.ensemble.spectral_density.compute_variate_coeffs(
                closed_eigenvalues
            )
            variate_cdfs.append(factory.interpolators_from_coeffs(coefficients)[0])

        result = simulation.execute()

        self.assertEqual(simulation.compound.rng_state, control.rng_state)
        self.assertEqual(result.context.rng["initial_state"], initial_rng_state)
        self.assertEqual(result.context.rng["final_state"], control.rng_state)
        for energy_index, energy in enumerate(simulation.energies):
            raw_samples = [sample[energy_index] for sample in samples]
            raw_histogram = result.raw[energy_index].histogram
            np.testing.assert_array_equal(
                raw_histogram.counts,
                histogram_counts(raw_samples, raw_histogram.bins),
            )

            average_samples = [
                unfold_delay_values(
                    sample[energy_index],
                    energy=float(energy),
                    cdf=average_cdf,
                    dimension=control.ensemble.dimension,
                )
                for sample in samples
            ]
            average_histogram = (
                result.average_by_degree[0].by_energy[energy_index].histogram
            )
            np.testing.assert_array_equal(
                average_histogram.counts,
                histogram_counts(average_samples, average_histogram.bins),
            )

            variate_samples = [
                unfold_delay_values(
                    sample[energy_index],
                    energy=float(energy),
                    cdf=cdf,
                    dimension=control.ensemble.dimension,
                )
                for sample, cdf in zip(samples, variate_cdfs, strict=True)
            ]
            variate_histogram = (
                result.variate_by_degree[0].by_energy[energy_index].histogram
            )
            np.testing.assert_array_equal(
                variate_histogram.counts,
                histogram_counts(variate_samples, variate_histogram.bins),
            )

    def test_persistence_plotting_and_run_consumers(self) -> None:
        simulation = TimeDelayStatisticsSimulation(
            compound=build_compound(seed=81),
            realizs=1,
            energies=(0.123456, 0.123457),
            request=TimeDelayStatisticsRequest(unfolding_modes=("raw", "weight")),
        )
        result = simulation.execute()
        completed_rng_state = deepcopy(simulation.compound.rng_state)

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = save_time_delay_statistics_result(result, out_dir=tmp_dir)
            restored = load_time_delay_statistics_result(run_dir)
            np.testing.assert_array_equal(
                restored.raw[0].histogram.counts,
                result.raw[0].histogram.counts,
            )
            np.testing.assert_array_equal(
                restored.energies,
                np.array([0.123456, 0.123457]),
            )

            with (
                patch.object(TimeDelayHistogramPlot, "plot", autospec=True) as raw_plot,
                patch.object(
                    UnfoldedTimeDelayHistogramPlot,
                    "plot",
                    autospec=True,
                ) as unfolded_plot,
            ):
                plot_time_delay_statistics_result(
                    result,
                    out_dir=tmp_dir,
                    modes="raw",
                    energy_indices=1,
                )
            raw_plot.assert_called_once()
            unfolded_plot.assert_not_called()
            self.assertIs(raw_plot.call_args.args[0].data, result.raw[1].histogram)
        self.assertEqual(simulation.compound.rng_state, completed_rng_state)

        data_only = TimeDelayStatisticsSimulation(
            compound=build_compound(seed=82),
            realizs=1,
            request=TimeDelayStatisticsRequest(unfolding_modes=("raw",)),
        )
        with patch("rmtpy.simulations.persistence.save_run") as save_result:
            run_result = data_only.execute()
        self.assertIs(run_result, data_only.result)
        save_result.assert_not_called()

    def test_request_uses_only_canonical_modes(self) -> None:
        request = TimeDelayStatisticsRequest(
            unfolding_modes=("variate", "raw"),
            degrees=(2,),
        )
        self.assertEqual(request.unfolding_modes, ("raw", "variate"))
        for legacy_name in ("wgt", "avg", "var"):
            with self.subTest(legacy_name=legacy_name), self.assertRaises(ValueError):
                TimeDelayStatisticsRequest(unfolding_modes=(legacy_name,))


if __name__ == "__main__":
    unittest.main()
