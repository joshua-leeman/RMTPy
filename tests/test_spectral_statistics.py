import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from typing import cast
from unittest.mock import patch

import attrs
import numpy as np

from rmtpy.conversion import unwrap_json_value
from rmtpy.ensembles import GOE
from rmtpy.simulations.base_simulation import ExecutionState
from rmtpy.simulations.histograms import Histogram
from rmtpy.simulations.spectral_statistics import (
    SpectralStatisticsSimulation,
    load_spectral_statistics_simulation,
    plot_spectral_statistics_simulation,
    run_spectral_statistics_simulation,
)
from rmtpy.simulations.spectral_statistics.nn_spacings_histogram import (
    SpacingsHistogram,
    SpacingsHistogramPlot,
    UnfoldedSpacingsHistogramPlot,
)
from rmtpy.simulations.spectral_statistics.spectral_coefficients_histogram import (
    SpectralCoefficientsHistogram,
    SpectralCoefficientsHistogramPlot,
)
from rmtpy.simulations.spectral_statistics.spectral_form_factors import (
    FormFactorsData,
    FormFactorsPlot,
    UnfoldedFormFactorsPlot,
)
from rmtpy.simulations.spectral_statistics.spectral_histogram import (
    SpectralHistogram,
    SpectralHistogramPlot,
    UnfoldedSpectralHistogramPlot,
)
from rmtpy.simulations.statistics import nearest_neighbor_spacings
from rmtpy.simulations.unfolding import TruncatedPolynomialCDFFactory, unfold_values


def build_ensemble(*, max_degree: int = 0, seed: int = 123) -> GOE:
    return GOE(
        num_majoranas=4,
        max_spectral_polynomial_degree=max_degree,
        seed=seed,
    )


def histogram_counts(
    samples: tuple[np.ndarray[tuple[int], np.dtype[np.floating]], ...],
    *,
    bins: np.ndarray[tuple[int], np.dtype[np.floating]],
) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
    counts = np.zeros(len(bins) - 1, dtype=np.int64)
    for sample in samples:
        indices = np.searchsorted(bins, sample, side="right") - 1
        valid = (indices >= 0) & (indices < len(counts))
        np.add.at(counts, indices[valid], 1)

    return counts


def form_factor_moments(
    samples: tuple[np.ndarray[tuple[int], np.dtype[np.floating]], ...],
    *,
    times: np.ndarray[tuple[int], np.dtype[np.floating]],
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.complexfloating]],
    np.ndarray[tuple[int], np.dtype[np.floating]],
]:
    first_moment = np.zeros(len(times), dtype=np.complex128)
    second_moment = np.zeros(len(times), dtype=np.float64)
    for sample in samples:
        contribution = np.sum(np.exp(-1j * np.outer(sample, times)), axis=0) / len(sample)
        first_moment += contribution
        second_moment += np.abs(contribution) ** 2

    return first_moment, second_moment


class SpectralStatisticsTests(unittest.TestCase):
    def test_buffer_schema_filenames_metadata_and_iteration_order(self) -> None:
        expected_counts = {0: 6, 2: 20}
        for max_degree, expected_count in expected_counts.items():
            with self.subTest(max_degree=max_degree):
                simulation = SpectralStatisticsSimulation(
                    ensemble=build_ensemble(max_degree=max_degree),
                    realizs=1,
                )

                coefficient_buffers = tuple(simulation.coefficient_buffers)
                averaged_buffers = tuple(simulation.ave_unfolded_buffers)
                variate_buffers = tuple(simulation.var_unfolded_buffers)
                data = tuple(simulation)

                self.assertEqual(len(data), expected_count)
                self.assertEqual(
                    tuple(histogram._file_name for histogram in coefficient_buffers),
                    tuple(
                        f"spectral_coeff_{degree}_histogram"
                        for degree in range(1, max_degree + 1)
                    ),
                )
                self.assertEqual(
                    tuple(item._file_name for item in simulation.raw_buffers),
                    (
                        "spectral_histogram",
                        "spacings_histogram",
                        "spectral_form_factors",
                    ),
                )
                self.assertEqual(
                    tuple(item._file_name for item in simulation.wgt_unfolded_buffers),
                    (
                        "spectral_histogram_weight_unfolded",
                        "spacings_histogram_weight_unfolded",
                        "spectral_form_factors_weight_unfolded",
                    ),
                )
                self.assertEqual(
                    tuple(buffer.polynomial_degree for buffer in averaged_buffers),
                    tuple(range(1, max_degree + 1)),
                )
                self.assertEqual(
                    tuple(buffer.polynomial_degree for buffer in variate_buffers),
                    tuple(range(1, max_degree + 1)),
                )

                for degree, histogram in enumerate(coefficient_buffers, start=1):
                    self.assertIsInstance(histogram, SpectralCoefficientsHistogram)
                    self.assertEqual(
                        histogram.metadata,
                        {"degree": degree, "unfolding": "raw"},
                    )

                self.assertIsInstance(
                    simulation.raw_buffers.levels,
                    SpectralHistogram,
                )
                self.assertIsInstance(
                    simulation.raw_buffers.nn_spacings,
                    SpacingsHistogram,
                )
                self.assertIsInstance(
                    simulation.raw_buffers.form_factors,
                    FormFactorsData,
                )
                self.assertEqual(
                    simulation.raw_buffers.levels.metadata,
                    {"unfolding": "raw"},
                )
                self.assertIn(
                    "global_mean_spacing",
                    simulation.raw_buffers.nn_spacings.metadata,
                )
                self.assertEqual(
                    simulation.raw_buffers.form_factors.metadata,
                    {"unfolding": "raw"},
                )

                if max_degree:
                    self.assertEqual(
                        tuple(item._file_name for item in averaged_buffers[0]),
                        (
                            "spectral_histogram_averaged_unfolded_degree_1",
                            "spacings_histogram_averaged_unfolded_degree_1",
                            "spectral_form_factors_averaged_unfolded_degree_1",
                        ),
                    )
                    self.assertEqual(
                        tuple(item._file_name for item in variate_buffers[0]),
                        (
                            "spectral_histogram_variate_unfolded_degree_1",
                            "spacings_histogram_variate_unfolded_degree_1",
                            "spectral_form_factors_variate_unfolded_degree_1",
                        ),
                    )
                    for buffers in averaged_buffers:
                        for item in buffers:
                            self.assertEqual(
                                item.metadata["polynomial_degree"],
                                buffers.polynomial_degree,
                            )
                            self.assertEqual(item.metadata["unfolding"], "averaged")

    def test_seeded_accumulation_unfolding_and_finalization_match_control(
        self,
    ) -> None:
        simulation = SpectralStatisticsSimulation(
            ensemble=build_ensemble(max_degree=2, seed=314159),
            realizs=2,
        )
        control = build_ensemble(max_degree=2, seed=314159)
        initial_rng_state = deepcopy(simulation.ensemble.rng_state)

        cdf_factory = TruncatedPolynomialCDFFactory(
            density=control.spectral_density,
            degrees=(1, 2),
            density_name="spectral",
        )
        average_cdfs = cdf_factory.average_interpolators()
        eigenvalue_samples = tuple(control.eigvals_stream(realizs=2))
        coefficient_samples = tuple(
            control.spectral_density.compute_variate_coeffs(eigenvalues)
            for eigenvalues in eigenvalue_samples
        )

        returned = simulation.execute()

        self.assertIsNone(returned)
        self.assertEqual(simulation.execution_state, ExecutionState.COMPLETE)
        self.assertEqual(simulation.ensemble.rng_state, control.rng_state)
        self.assertEqual(simulation.manifest.rng["initial_state"], initial_rng_state)
        self.assertEqual(simulation.manifest.rng["final_state"], control.rng_state)

        calibration = cast(
            dict[str, object],
            simulation.manifest.execution["calibration"],
        )
        self.assertEqual(calibration["density"], "spectral")
        self.assertEqual(calibration["timing"], "cached_during_execution")

        for degree, histogram in enumerate(simulation.coefficient_buffers, start=1):
            expected_samples = tuple(
                coefficients[degree : degree + 1] for coefficients in coefficient_samples
            )
            np.testing.assert_array_equal(
                histogram.counts,
                histogram_counts(expected_samples, bins=histogram.bins),
            )

        np.testing.assert_array_equal(
            simulation.raw_buffers.levels.counts,
            histogram_counts(
                eigenvalue_samples,
                bins=simulation.raw_buffers.levels.bins,
            ),
        )
        raw_spacing_samples = tuple(
            nearest_neighbor_spacings(eigenvalues) for eigenvalues in eigenvalue_samples
        )
        np.testing.assert_array_equal(
            simulation.raw_buffers.nn_spacings.counts,
            histogram_counts(
                raw_spacing_samples,
                bins=simulation.raw_buffers.nn_spacings.bins,
            ),
        )

        weight_samples = tuple(
            unfold_values(
                eigenvalues,
                cdf=control.spectral_density.weight_cdf,
                dimension=control.dimension,
            )
            for eigenvalues in eigenvalue_samples
        )
        np.testing.assert_array_equal(
            simulation.wgt_unfolded_buffers.levels.counts,
            histogram_counts(
                weight_samples,
                bins=simulation.wgt_unfolded_buffers.levels.bins,
            ),
        )

        for cdf, buffers in zip(
            average_cdfs,
            simulation.ave_unfolded_buffers,
            strict=True,
        ):
            expected_samples = tuple(
                unfold_values(
                    eigenvalues,
                    cdf=cdf,
                    dimension=control.dimension,
                )
                for eigenvalues in eigenvalue_samples
            )
            np.testing.assert_array_equal(
                buffers.levels.counts,
                histogram_counts(expected_samples, bins=buffers.levels.bins),
            )

        variate_samples: list[list[np.ndarray[tuple[int], np.dtype[np.floating]]]] = [
            [],
            [],
        ]
        for eigenvalues, coefficients in zip(
            eigenvalue_samples,
            coefficient_samples,
            strict=True,
        ):
            variate_cdfs = cdf_factory.interpolators_from_coeffs(coefficients)
            for index, cdf in enumerate(variate_cdfs):
                variate_samples[index].append(
                    unfold_values(
                        eigenvalues,
                        cdf=cdf,
                        dimension=control.dimension,
                    )
                )

        for buffers, expected_samples in zip(
            simulation.var_unfolded_buffers,
            variate_samples,
            strict=True,
        ):
            np.testing.assert_array_equal(
                buffers.levels.counts,
                histogram_counts(tuple(expected_samples), bins=buffers.levels.bins),
            )

        raw_form_factors = simulation.raw_buffers.form_factors
        expected_first_moment, expected_second_moment = form_factor_moments(
            eigenvalue_samples,
            times=raw_form_factors.times,
        )
        np.testing.assert_allclose(
            raw_form_factors.first_moment,
            expected_first_moment,
        )
        np.testing.assert_allclose(
            raw_form_factors.second_moment,
            expected_second_moment,
        )
        np.testing.assert_allclose(
            raw_form_factors.form_factor,
            expected_second_moment / simulation.realizs,
        )
        np.testing.assert_allclose(
            raw_form_factors.connected_form_factor,
            raw_form_factors.form_factor
            - np.abs(expected_first_moment / simulation.realizs) ** 2,
            atol=1e-15,
        )

        for data in simulation:
            if isinstance(data, Histogram) and np.sum(data.counts):
                np.testing.assert_allclose(
                    data.histogram,
                    data.counts / (np.sum(data.counts) * np.diff(data.bins)),
                )

    def test_precomputed_calibration_is_recorded_without_resampling(self) -> None:
        ensemble = build_ensemble(max_degree=2, seed=24)
        average_coefficients = ensemble.spectral_density.average_coeffs.copy()
        simulation = SpectralStatisticsSimulation(ensemble=ensemble, realizs=1)
        execution_start_state = deepcopy(ensemble.rng_state)

        simulation.execute()

        calibration = cast(
            dict[str, object],
            simulation.manifest.execution["calibration"],
        )
        self.assertEqual(calibration["timing"], "previously_cached")
        np.testing.assert_array_equal(
            unwrap_json_value(calibration["average_coefficients"]),
            average_coefficients,
        )
        self.assertEqual(
            simulation.manifest.rng["initial_state"],
            execution_start_state,
        )

    def test_failed_execution_is_terminal(self) -> None:
        simulation = SpectralStatisticsSimulation(
            ensemble=build_ensemble(),
            realizs=1,
        )
        with (
            patch.object(
                SpectralStatisticsSimulation,
                "_realize_spectral_statistics",
                side_effect=RuntimeError("expected failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "expected failure"),
        ):
            simulation.execute()

        self.assertEqual(simulation.execution_state, ExecutionState.FAILED)
        with self.assertRaisesRegex(RuntimeError, "only once"):
            simulation.execute()
        with self.assertRaisesRegex(RuntimeError, "only after execution"):
            simulation.save()
        with self.assertRaisesRegex(RuntimeError, "only after execution"):
            simulation.plot(Path("unused"))

    def test_save_load_plot_dispatch_and_archive_validation(self) -> None:
        simulation = SpectralStatisticsSimulation(
            ensemble=build_ensemble(max_degree=1, seed=77),
            realizs=1,
        )
        simulation.execute()
        completed_rng_state = deepcopy(simulation.ensemble.rng_state)

        with tempfile.TemporaryDirectory() as temporary_directory:
            destination_directory = simulation.save(temporary_directory)
            restored_simulation = load_spectral_statistics_simulation(
                directory=destination_directory
            )

            self.assertEqual(
                restored_simulation.execution_state,
                ExecutionState.COMPLETE,
            )
            self.assertEqual(
                tuple(type(data) for data in restored_simulation),
                tuple(type(data) for data in simulation),
            )
            for restored_data, original_data in zip(
                restored_simulation,
                simulation,
                strict=True,
            ):
                self.assertEqual(restored_data.metadata, original_data.metadata)
                for field in attrs.fields(type(original_data)):
                    restored_value = getattr(restored_data, field.name)
                    original_value = getattr(original_data, field.name)
                    if isinstance(original_value, np.ndarray):
                        np.testing.assert_array_equal(restored_value, original_value)

            np.testing.assert_array_equal(
                restored_simulation.ensemble.spectral_density.average_coeffs,
                simulation.ensemble.spectral_density.average_coeffs,
            )

            with (
                patch.object(
                    SpectralCoefficientsHistogramPlot,
                    "plot",
                    autospec=True,
                ) as coefficient_plot,
                patch.object(
                    SpectralHistogramPlot,
                    "plot",
                    autospec=True,
                ) as raw_spectral_plot,
                patch.object(
                    UnfoldedSpectralHistogramPlot,
                    "plot",
                    autospec=True,
                ) as unfolded_spectral_plot,
                patch.object(
                    SpacingsHistogramPlot,
                    "plot",
                    autospec=True,
                ) as raw_spacings_plot,
                patch.object(
                    UnfoldedSpacingsHistogramPlot,
                    "plot",
                    autospec=True,
                ) as unfolded_spacings_plot,
                patch.object(
                    FormFactorsPlot,
                    "plot",
                    autospec=True,
                ) as raw_form_factors_plot,
                patch.object(
                    UnfoldedFormFactorsPlot,
                    "plot",
                    autospec=True,
                ) as unfolded_form_factors_plot,
            ):
                plot_spectral_statistics_simulation(directory=destination_directory)

            coefficient_plot.assert_called_once()
            raw_spectral_plot.assert_called_once()
            raw_spacings_plot.assert_called_once()
            raw_form_factors_plot.assert_called_once()
            self.assertEqual(unfolded_spectral_plot.call_count, 3)
            self.assertEqual(unfolded_spacings_plot.call_count, 3)
            self.assertEqual(unfolded_form_factors_plot.call_count, 3)
            self.assertIsNotNone(raw_spectral_plot.call_args.args[0].context)

            manifest_path = destination_directory / "manifest.json"
            original_manifest_text = manifest_path.read_text(encoding="utf-8")
            malformed_manifest = json.loads(original_manifest_text)
            malformed_manifest["execution"]["calibration"]["average_coefficients"] = []
            manifest_path.write_text(
                json.dumps(malformed_manifest, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "invalid shape"):
                load_spectral_statistics_simulation(directory=destination_directory)
            manifest_path.write_text(original_manifest_text, encoding="utf-8")

            unexpected = SpectralHistogram(
                _file_name="unexpected_spectral_histogram",
                support=(-1.0, 1.0),
            )
            unexpected.save(directory=destination_directory)
            with self.assertRaisesRegex(ValueError, "not part of the simulation"):
                load_spectral_statistics_simulation(directory=destination_directory)
            (destination_directory / unexpected.to_path).unlink()

            missing_data_path = (
                destination_directory / next(iter(restored_simulation)).to_path
            )
            missing_data_path.unlink()
            with self.assertRaisesRegex(ValueError, "Saved data .* is missing"):
                load_spectral_statistics_simulation(directory=destination_directory)
            with self.assertRaisesRegex(ValueError, "Saved data .* is missing"):
                restored_simulation.plot(destination_directory)

        self.assertEqual(simulation.ensemble.rng_state, completed_rng_state)

    def test_plot_configuration_uses_a_detached_ensemble(self) -> None:
        simulation = SpectralStatisticsSimulation(
            ensemble=build_ensemble(seed=902),
            realizs=1,
        )
        simulation.execute()
        completed_rng_state = deepcopy(simulation.ensemble.rng_state)

        plot = SpectralHistogramPlot(
            data=simulation.raw_buffers.levels,
            context=simulation.manifest,
        )
        plot.set_derived_attributes()

        self.assertIsNot(plot.ensemble, simulation.ensemble)
        self.assertEqual(simulation.ensemble.rng_state, completed_rng_state)

    def test_run_helper_executes_saves_reloads_and_dispatches_plots(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            patch.object(SpectralHistogramPlot, "plot", autospec=True),
            patch.object(UnfoldedSpectralHistogramPlot, "plot", autospec=True),
            patch.object(SpacingsHistogramPlot, "plot", autospec=True),
            patch.object(UnfoldedSpacingsHistogramPlot, "plot", autospec=True),
            patch.object(FormFactorsPlot, "plot", autospec=True),
            patch.object(UnfoldedFormFactorsPlot, "plot", autospec=True),
        ):
            simulation = run_spectral_statistics_simulation(
                ensemble=build_ensemble(seed=311),
                realizs=1,
                directory=temporary_directory,
            )

            self.assertEqual(simulation.execution_state, ExecutionState.COMPLETE)
            completion_time = simulation.manifest.execution["execution_time"]
            destination_directory = (
                Path(temporary_directory) / simulation.to_path / str(completion_time)
            )
            self.assertTrue((destination_directory / "manifest.json").is_file())
            self.assertEqual(
                len(tuple(destination_directory.rglob("*_data.npz"))),
                len(tuple(simulation)),
            )


if __name__ == "__main__":
    unittest.main()
