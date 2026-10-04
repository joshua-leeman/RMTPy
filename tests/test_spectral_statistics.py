import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import attrs
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from rmtpy.conversion import AttrsFields, unwrap_json_value
from rmtpy.ensembles import GOE
from rmtpy.simulations.base_data import Data
from rmtpy.simulations.base_plot import (
    ENSEMBLE_AVERAGED_CURVE_WIDTH,
    SINGLE_REALIZATION_CURVE_WIDTH,
)
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
from rmtpy.simulations.statistics import (
    COEFFICIENT_GRID_POLICY,
    nearest_neighbor_spacings,
)
from rmtpy.simulations.unfolding import TruncatedPolynomialCDFFactory, unfold_values
from tests.support import (
    ArchiveArray,
    ComplexArray,
    FloatVector,
    archive_fields,
    json_mapping,
    manifest_section,
    mock_argument,
)


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
    np.ndarray[tuple[int], np.dtype[np.floating]],
]:
    first_moment = np.zeros(len(times), dtype=np.complex128)
    second_moment = np.zeros(len(times), dtype=np.float64)
    single_realization_form_factor = np.zeros(len(times), dtype=np.float64)
    for index, sample in enumerate(samples):
        contribution = cast(
            ComplexArray,
            np.sum(np.exp(-1j * np.outer(sample, times)), axis=0) / len(sample),
        )
        first_moment += contribution
        second_moment += np.abs(contribution) ** 2
        if index == 0:
            single_realization_form_factor[:] = np.abs(contribution) ** 2

    return first_moment, second_moment, single_realization_form_factor


class SpectralStatisticsTests(unittest.TestCase):
    def test_form_factor_curves_and_legends_use_accessible_shared_style(self) -> None:
        simulation = SpectralStatisticsSimulation(
            ensemble=build_ensemble(),
            realizs=1,
        )
        plot_cases = (
            (
                FormFactorsPlot(
                    data=simulation.raw_buffers.form_factors,
                    context=simulation.manifest,
                ),
                ("#0072B2", "#D54300", "#009E73"),
                ("-", "-", "-"),
            ),
            (
                UnfoldedFormFactorsPlot(
                    data=simulation.wgt_unfolded_buffers.form_factors,
                    context=simulation.manifest,
                ),
                ("#0072B2", "#D54300", "Black", "#009E73"),
                ("-", "-", ":", "-"),
            ),
        )

        for plot, expected_colors, expected_styles in plot_cases:
            with self.subTest(plot=type(plot).__name__):
                with patch.object(plot, "finish_plot"):
                    plot.plot(Path("unused"))

                try:
                    self.assertEqual(
                        tuple(line.get_color() for line in plot.ax.lines),
                        expected_colors,
                    )
                    self.assertEqual(
                        tuple(line.get_linewidth() for line in plot.ax.lines),
                        (ENSEMBLE_AVERAGED_CURVE_WIDTH,) * (len(expected_colors) - 1)
                        + (SINGLE_REALIZATION_CURVE_WIDTH,),
                    )
                    self.assertEqual(
                        tuple(line.get_linestyle() for line in plot.ax.lines),
                        expected_styles,
                    )
                    self.assertEqual(
                        tuple(
                            handle.get_color()
                            for handle in cast(tuple[Line2D, ...], plot.legend.handles)
                        ),
                        expected_colors,
                    )
                    self.assertEqual(
                        tuple(
                            handle.get_linewidth()
                            for handle in cast(tuple[Line2D, ...], plot.legend.handles)
                        ),
                        (ENSEMBLE_AVERAGED_CURVE_WIDTH,) * (len(expected_colors) - 1)
                        + (SINGLE_REALIZATION_CURVE_WIDTH,),
                    )
                    self.assertEqual(
                        tuple(
                            handle.get_linestyle()
                            for handle in cast(tuple[Line2D, ...], plot.legend.handles)
                        ),
                        expected_styles,
                    )
                    self.assertEqual(plot.ax.lines[-1].get_alpha(), 1.0)
                    self.assertEqual(plot.ax.lines[-1].get_zorder(), 1)
                    np.testing.assert_array_equal(
                        plot.ax.lines[-1].get_ydata(),
                        cast(FormFactorsData, plot.data).single_realization_form_factor,
                    )
                    self.assertEqual(
                        plot.legend.labels[-1],
                        plot.single_sff_legend,
                    )
                finally:
                    plt.close(plot.fig)

    def test_all_plot_titles_use_compact_subjects_and_ordered_metadata(self) -> None:
        simulation = SpectralStatisticsSimulation(
            ensemble=build_ensemble(max_degree=2),
            realizs=1,
        )
        ensemble = simulation.ensemble.to_latex

        raw_cases = (
            (
                SpectralHistogramPlot,
                simulation.raw_buffers.levels,
                f"Spectral PDF: {ensemble}",
            ),
            (
                SpacingsHistogramPlot,
                simulation.raw_buffers.nn_spacings,
                f"NNS PDF: {ensemble}",
            ),
            (
                FormFactorsPlot,
                simulation.raw_buffers.form_factors,
                f"Spectral Form Factors: {ensemble}",
            ),
            (
                SpectralCoefficientsHistogramPlot,
                tuple(simulation.coefficient_buffers)[0],
                f"Spectral Coefficients: {ensemble}",
            ),
        )
        for plot_cls, data, expected_title in raw_cases:
            with self.subTest(title=expected_title):
                plot = plot_cls(data=data, context=simulation.manifest)
                plot.set_derived_attributes()
                self.assertEqual(plot.axes.title, expected_title)
                self.assertNotIn("\n", plot.axes.title)

        unfolded_cases = (
            (
                simulation.wgt_unfolded_buffers,
                "Wgt-unfolded",
            ),
            (
                tuple(simulation.ave_unfolded_buffers)[1],
                r"Ave(2)-unfolded",
            ),
            (
                tuple(simulation.var_unfolded_buffers)[0],
                r"Var(1)-unfolded",
            ),
        )
        plot_cases = (
            (UnfoldedSpectralHistogramPlot, "levels", "Spectral PDF"),
            (UnfoldedSpacingsHistogramPlot, "nn_spacings", "NNS PDF"),
            (
                UnfoldedFormFactorsPlot,
                "form_factors",
                "Spectral Form Factors",
            ),
        )
        for buffers, unfolding in unfolded_cases:
            for plot_cls, data_name, subject in plot_cases:
                expected_title = f"{unfolding} {subject}: {ensemble}"
                with self.subTest(title=expected_title):
                    plot = plot_cls(
                        data=cast(Data, getattr(buffers, data_name)),
                        context=simulation.manifest,
                    )
                    plot.set_derived_attributes()
                    self.assertEqual(plot.axes.title, expected_title)
                    self.assertNotIn("\n", plot.axes.title)

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
                        {
                            "degree": degree,
                            "unfolding": "raw",
                            "grid_policy": COEFFICIENT_GRID_POLICY,
                            "dimension": simulation.ensemble.dimension,
                        },
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
                            "spectral_histogram_average_unfolded_degree_1",
                            "spacings_histogram_average_unfolded_degree_1",
                            "spectral_form_factors_average_unfolded_degree_1",
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
                            self.assertEqual(item.metadata["unfolding"], "average")

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
            expected_coefficients = np.concatenate(expected_samples)

            self.assertEqual(histogram.realizs, simulation.realizs)
            np.testing.assert_array_equal(
                histogram.counts,
                histogram_counts(expected_samples, bins=histogram.bins),
            )
            self.assertEqual(
                np.sum(histogram.counts) + histogram.underflow + histogram.overflow,
                simulation.realizs,
            )
            self.assertEqual(
                histogram.underflow,
                int(
                    np.count_nonzero(
                        expected_coefficients < cast(np.floating, histogram.bins[0])
                    )
                ),
            )
            self.assertEqual(
                histogram.overflow,
                int(
                    np.count_nonzero(
                        expected_coefficients >= cast(np.floating, histogram.bins[-1])
                    )
                ),
            )
            self.assertAlmostEqual(
                np.sum(histogram.histogram * np.diff(histogram.bins)),
                1.0 if np.sum(histogram.counts) else 0.0,
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
        _, _, expected_single_form_factor = form_factor_moments(
            weight_samples,
            times=simulation.wgt_unfolded_buffers.form_factors.times,
        )
        np.testing.assert_allclose(
            simulation.wgt_unfolded_buffers.form_factors.single_realization_form_factor,
            expected_single_form_factor,
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
            _, _, expected_single_form_factor = form_factor_moments(
                expected_samples,
                times=buffers.form_factors.times,
            )
            np.testing.assert_allclose(
                buffers.form_factors.single_realization_form_factor,
                expected_single_form_factor,
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
            _, _, expected_single_form_factor = form_factor_moments(
                tuple(expected_samples),
                times=buffers.form_factors.times,
            )
            np.testing.assert_allclose(
                buffers.form_factors.single_realization_form_factor,
                expected_single_form_factor,
            )

        raw_form_factors = simulation.raw_buffers.form_factors
        (
            expected_first_moment,
            expected_second_moment,
            expected_single_form_factor,
        ) = form_factor_moments(
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
            raw_form_factors.single_realization_form_factor,
            expected_single_form_factor,
        )
        np.testing.assert_allclose(
            raw_form_factors.form_factor,
            expected_second_moment / simulation.realizs,
        )
        np.testing.assert_allclose(
            raw_form_factors.connected_form_factor,
            (
                raw_form_factors.form_factor
                - np.abs(expected_first_moment / simulation.realizs) ** 2
            )
            * (simulation.realizs / (simulation.realizs - 1)),
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
            _ = simulation.save()
        with self.assertRaisesRegex(RuntimeError, "only after execution"):
            simulation.plot(Path("unused"))

    def test_save_load_plot_dispatch_and_archive_validation(self) -> None:
        simulation = SpectralStatisticsSimulation(
            ensemble=build_ensemble(max_degree=2, seed=314159),
            realizs=2,
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
                for field in cast(AttrsFields, attrs.fields(type(original_data))):
                    restored_value = cast(object, getattr(restored_data, field.name))
                    original_value = cast(object, getattr(original_data, field.name))
                    if isinstance(original_value, np.ndarray):
                        np.testing.assert_array_equal(
                            cast(ArchiveArray, restored_value),
                            cast(ArchiveArray, original_value),
                        )

            np.testing.assert_array_equal(
                restored_simulation.ensemble.spectral_density.average_coeffs,
                simulation.ensemble.spectral_density.average_coeffs,
            )

            native_coefficient_plots = tuple(
                SpectralCoefficientsHistogramPlot(
                    data=histogram,
                    context=restored_simulation.manifest,
                )
                for histogram in restored_simulation.coefficient_buffers
            )
            for plot in native_coefficient_plots:
                plot.set_derived_attributes()

            widest_plot = max(
                native_coefficient_plots,
                key=lambda plot: plot.xlim[1] - plot.xlim[0],
            )
            horizontal_padding = 0.5 * (
                widest_plot.axes.xticks[1] - widest_plot.axes.xticks[0]
            )
            self.assertAlmostEqual(
                widest_plot.xlim[0],
                widest_plot.axes.xticks[0] - horizontal_padding,
            )
            self.assertAlmostEqual(
                widest_plot.xlim[1],
                widest_plot.axes.xticks[-1] + horizontal_padding,
            )
            self.assertAlmostEqual(widest_plot.xlim[0], -widest_plot.xlim[1])

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

            self.assertEqual(coefficient_plot.call_count, 2)
            raw_spectral_plot.assert_called_once()
            raw_spacings_plot.assert_called_once()
            raw_form_factors_plot.assert_called_once()
            self.assertEqual(unfolded_spectral_plot.call_count, 5)
            self.assertEqual(unfolded_spacings_plot.call_count, 5)
            self.assertEqual(unfolded_form_factors_plot.call_count, 5)
            self.assertIsNotNone(
                mock_argument(raw_spectral_plot, 0, SpectralHistogramPlot).context
            )

            shared_coefficient_plots = tuple(
                cast(SpectralCoefficientsHistogramPlot, call.args[0])
                for call in coefficient_plot.call_args_list
            )
            self.assertEqual(
                tuple(plot.axes.xlabel for plot in shared_coefficient_plots),
                (r"$c_{1}$", r"$c_{2}$"),
            )
            for native_plot, shared_plot in zip(
                native_coefficient_plots,
                shared_coefficient_plots,
                strict=True,
            ):
                self.assertEqual(shared_plot.xlim, widest_plot.xlim)
                self.assertEqual(shared_plot.axes.xticks, widest_plot.axes.xticks)
                self.assertEqual(
                    shared_plot.axes.xticks_minor,
                    widest_plot.axes.xticks_minor,
                )
                self.assertEqual(
                    shared_plot.axes.xtick_labels,
                    widest_plot.axes.xtick_labels,
                )

                self.assertEqual(shared_plot.ylim, native_plot.ylim)
                self.assertEqual(shared_plot.axes.yticks, native_plot.axes.yticks)
                self.assertEqual(
                    shared_plot.axes.yticks_minor,
                    native_plot.axes.yticks_minor,
                )
                self.assertEqual(
                    shared_plot.axes.ytick_labels,
                    native_plot.axes.ytick_labels,
                )

            manifest_path = destination_directory / "manifest.json"
            original_manifest_text = manifest_path.read_text(encoding="utf-8")
            malformed_manifest = json_mapping(original_manifest_text)
            manifest_section(malformed_manifest, "execution", "calibration")[
                "average_coefficients"
            ] = []
            _ = manifest_path.write_text(
                json.dumps(malformed_manifest, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "invalid shape"):
                _ = load_spectral_statistics_simulation(directory=destination_directory)
            _ = manifest_path.write_text(original_manifest_text, encoding="utf-8")

            form_factors_path = (
                destination_directory
                / restored_simulation.raw_buffers.form_factors.to_path
            )
            form_factors_payload = archive_fields(form_factors_path)
            single_form_factor = form_factors_payload["single_realization_form_factor"]
            form_factors_payload["single_realization_form_factor"] = single_form_factor[
                :-1
            ]
            np.savez(form_factors_path, allow_pickle=False, **form_factors_payload)
            with self.assertRaisesRegex(ValueError, "does not match"):
                _ = load_spectral_statistics_simulation(directory=destination_directory)

            _ = form_factors_payload.pop("single_realization_form_factor")
            np.savez(form_factors_path, allow_pickle=False, **form_factors_payload)
            legacy_simulation = load_spectral_statistics_simulation(
                directory=destination_directory
            )
            self.assertFalse(
                legacy_simulation.raw_buffers.form_factors.single_realization_form_factor_available
            )

            form_factors_payload["single_realization_form_factor"] = single_form_factor
            np.savez(form_factors_path, allow_pickle=False, **form_factors_payload)

            unexpected = SpectralHistogram(
                _file_name="unexpected_spectral_histogram",
                support=(-1.0, 1.0),
            )
            unexpected.save(directory=destination_directory)
            with self.assertRaisesRegex(ValueError, "not part of the simulation"):
                _ = load_spectral_statistics_simulation(directory=destination_directory)
            (destination_directory / unexpected.to_path).unlink()

            missing_data_path = (
                destination_directory / next(iter(restored_simulation)).to_path
            )
            missing_data_path.unlink()
            with self.assertRaisesRegex(ValueError, "Saved data .* is missing"):
                _ = load_spectral_statistics_simulation(directory=destination_directory)
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

    def test_degree_zero_raw_spectral_plot_draws_polynomial_weight_pdf(
        self,
    ) -> None:
        simulation = SpectralStatisticsSimulation(
            ensemble=build_ensemble(max_degree=0, seed=271),
            realizs=2,
        )
        simulation.execute()
        self.assertEqual(simulation.manifest.execution["calibration"], {})

        plot = SpectralHistogramPlot(
            data=simulation.raw_buffers.levels,
            context=simulation.manifest,
        )
        plot.ax = MagicMock()
        with (
            patch.object(plot, "build_figure"),
            patch.object(plot, "draw_histogram"),
            patch.object(plot, "finish_plot"),
        ):
            plot.plot(Path("unused"))

        cast(MagicMock, plot.ax.plot).assert_called_once()
        energies = mock_argument(cast(MagicMock, plot.ax.plot), 0, np.ndarray)
        spectral_pdf = mock_argument(cast(MagicMock, plot.ax.plot), 1, np.ndarray)
        np.testing.assert_array_equal(
            energies,
            np.linspace(*plot.xlim, plot.num_points),
        )
        np.testing.assert_allclose(
            spectral_pdf,
            plot.ensemble.spectral_density.weight_pdf(energies),
        )
        self.assertEqual(
            plot.legend.labels,
            ("simulation", "polynomial weight"),
        )
        self.assertEqual(len(plot.legend.handles), 2)

    def test_coefficient_plot_frames_central_mass_without_discarding_outlier(
        self,
    ) -> None:
        samples = np.append(np.linspace(-0.1, 0.1, 1_000), 10.0)
        bins = np.histogram_bin_edges(samples, bins="fd")
        bins[-1] = np.nextafter(cast(np.floating, bins[-1]), np.inf)
        counts = np.histogram(samples, bins=bins)[0]
        histogram = SpectralCoefficientsHistogram(
            metadata={"degree": 1, "unfolding": "raw"},
            _file_name="spectral_coeff_1_histogram",
            support=(
                float(cast(np.floating, bins[0])),
                float(cast(np.floating, bins[-1])),
            ),
            num_bins=len(counts),
            bins=bins,
            counts=counts,
            realizs=len(samples),
        )
        histogram.compute_histogram()

        context = SpectralStatisticsSimulation(
            ensemble=build_ensemble(seed=23),
            realizs=1,
        ).manifest
        plot = SpectralCoefficientsHistogramPlot(data=histogram, context=context)
        plot.set_derived_attributes()

        self.assertEqual(np.sum(histogram.counts), len(samples))
        self.assertGreater(histogram.support[1], float(cast(np.floating, samples[-1])))
        self.assertGreaterEqual(
            plot.xlim[1], float(np.max(cast(FloatVector, samples[:-1])))
        )
        self.assertLess(plot.xlim[1], float(cast(np.floating, samples[-1])))
        self.assertEqual(plot.xlim[0], -plot.xlim[1])

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
    _ = unittest.main()
