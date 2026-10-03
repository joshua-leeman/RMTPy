import json
import tempfile
import unittest
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import attrs
import numpy as np
from matplotlib import pyplot as plt

from rmtpy.compounds import CompoundEnsemble
from rmtpy.ensembles import GOE
from rmtpy.simulations.base_simulation import ExecutionState
from rmtpy.simulations.resonance_statistics import (
    ResonanceStatisticsSimulation,
    load_resonance_statistics_simulation,
    plot_resonance_statistics_simulation,
    run_resonance_statistics_simulation,
)
from rmtpy.simulations.resonance_statistics.complex_energy_histogram import (
    ComplexEnergyHistogram,
    ComplexEnergyHistogramPlot,
    UnfoldedComplexEnergyHistogramPlot,
)
from rmtpy.simulations.resonance_statistics.resonance_coefficients_histogram import (
    ResonanceCoefficientsHistogram,
    ResonanceCoefficientsHistogramPlot,
)
from rmtpy.simulations.resonance_statistics.resonance_form_factors import (
    FormFactorsData,
    ResonanceFormFactorsPlot,
    UnfoldedResonanceFormFactorsPlot,
)
from rmtpy.simulations.resonance_statistics.resonance_histogram import (
    ResonanceHistogram,
    ResonanceHistogramPlot,
    UnfoldedResonanceHistogramPlot,
)
from rmtpy.simulations.resonance_statistics.resonance_spacing_histogram import (
    ResonanceSpacingHistogram,
    ResonanceSpacingHistogramPlot,
    UnfoldedResonanceSpacingHistogramPlot,
)
from rmtpy.simulations.resonance_statistics.width_histogram import (
    UnfoldedWidthHistogramPlot,
    WidthHistogram,
    WidthHistogramPlot,
)
from rmtpy.simulations.unfolding import (
    TruncatedPolynomialCDFFactory,
    unfold_values,
    unfold_widths,
)

RESONANCE_PLOT_CLASSES = (
    ResonanceCoefficientsHistogramPlot,
    ResonanceHistogramPlot,
    UnfoldedResonanceHistogramPlot,
    WidthHistogramPlot,
    UnfoldedWidthHistogramPlot,
    ResonanceSpacingHistogramPlot,
    UnfoldedResonanceSpacingHistogramPlot,
    UnfoldedComplexEnergyHistogramPlot,
    ComplexEnergyHistogramPlot,
    ResonanceFormFactorsPlot,
    UnfoldedResonanceFormFactorsPlot,
)


def build_compound(*, max_degree: int = 0, seed: int = 123) -> CompoundEnsemble:
    return CompoundEnsemble(
        ensemble=GOE(
            num_majoranas=4,
            max_spectral_polynomial_degree=max_degree,
            seed=seed,
        ),
        couplings=np.array([0.75, 1.25]),
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


def histogram2d_counts(
    x_values: np.ndarray[tuple[int], np.dtype[np.floating]],
    y_values: np.ndarray[tuple[int], np.dtype[np.floating]],
    *,
    histogram: ComplexEnergyHistogram,
) -> np.ndarray[tuple[int, int], np.dtype[np.int64]]:
    counts = np.zeros_like(histogram.counts)
    x_indices = np.searchsorted(histogram.x_bins, x_values, side="right") - 1
    y_indices = np.searchsorted(histogram.y_bins, y_values, side="right") - 1
    valid = (
        (x_indices >= 0)
        & (x_indices < counts.shape[0])
        & (y_indices >= 0)
        & (y_indices < counts.shape[1])
    )
    np.add.at(counts, (x_indices[valid], y_indices[valid]), 1)

    return counts


class ResonanceStatisticsTests(unittest.TestCase):
    def test_form_factor_curves_and_legends_use_accessible_shared_style(self) -> None:
        simulation = ResonanceStatisticsSimulation(
            compound=build_compound(),
            realizs=1,
        )
        plot_cases = (
            (
                ResonanceFormFactorsPlot(
                    data=simulation.raw_buffers.form_factors,
                    context=simulation.manifest,
                ),
                ("#0072B2", "#D55E00", "#009E73"),
                ("-", "-", "-"),
            ),
            (
                UnfoldedResonanceFormFactorsPlot(
                    data=simulation.wgt_unfolded_buffers.form_factors,
                    context=simulation.manifest,
                ),
                ("#0072B2", "#D55E00", "Black", "#009E73"),
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
                        (2.0,) * len(expected_colors),
                    )
                    self.assertEqual(
                        tuple(line.get_linestyle() for line in plot.ax.lines),
                        expected_styles,
                    )
                    self.assertEqual(
                        tuple(handle.get_color() for handle in plot.legend.handles),
                        expected_colors,
                    )
                    self.assertEqual(
                        tuple(handle.get_linewidth() for handle in plot.legend.handles),
                        (2.0,) * len(expected_colors),
                    )
                    self.assertEqual(
                        tuple(handle.get_linestyle() for handle in plot.legend.handles),
                        expected_styles,
                    )
                    self.assertEqual(plot.ax.lines[-1].get_alpha(), 1.0)
                    self.assertEqual(plot.ax.lines[-1].get_zorder(), 3)
                    np.testing.assert_array_equal(
                        plot.ax.lines[-1].get_ydata(),
                        plot.data.single_realization_form_factor,
                    )
                    self.assertEqual(
                        plot.legend.labels[-1],
                        plot.single_sff_legend,
                    )
                finally:
                    plt.close(plot.fig)

    def test_all_plot_titles_use_compact_subjects_and_ordered_metadata(self) -> None:
        simulation = ResonanceStatisticsSimulation(
            compound=build_compound(max_degree=2),
            realizs=1,
        )
        metadata = (
            simulation.compound.ensemble.to_latex
            + r", $N_\text{f} = {1}$, {$\alpha = {-0.6}$}"
        )
        raw_cases = (
            (
                ResonanceHistogramPlot,
                simulation.raw_buffers.resonance_centers,
                "Resonance Density",
            ),
            (
                WidthHistogramPlot,
                simulation.raw_buffers.resonance_widths,
                "Resonance Width Distribution",
            ),
            (
                ResonanceSpacingHistogramPlot,
                simulation.raw_buffers.nn_spacings,
                "Resonance Spacing Distribution",
            ),
            (
                ComplexEnergyHistogramPlot,
                simulation.raw_buffers.complex_energies,
                "Complex Resonances",
            ),
            (
                ResonanceFormFactorsPlot,
                simulation.raw_buffers.form_factors,
                "Resonance Form Factors",
            ),
            (
                ResonanceCoefficientsHistogramPlot,
                tuple(simulation.coefficient_buffers)[0],
                "Resonance Coefficients",
            ),
        )
        for plot_cls, data, subject in raw_cases:
            expected_title = f"{subject}: {metadata}"
            with self.subTest(title=expected_title):
                plot = plot_cls(data=data, context=simulation.manifest)
                plot.set_derived_attributes()
                self.assertEqual(plot.axes.title, expected_title)
                self.assertNotIn("\n", plot.axes.title)

        unfolded_cases = (
            (
                simulation.wgt_unfolded_buffers,
                "Weight-unfolded",
            ),
            (
                tuple(simulation.ave_unfolded_buffers)[1],
                r"Average-unfolded (deg = $2$)",
            ),
            (
                tuple(simulation.var_unfolded_buffers)[0],
                r"Variate-unfolded (deg = $1$)",
            ),
        )
        plot_cases = (
            (
                UnfoldedResonanceHistogramPlot,
                "resonance_centers",
                "Resonance Density",
            ),
            (
                UnfoldedWidthHistogramPlot,
                "resonance_widths",
                "Resonance Width Distribution",
            ),
            (
                UnfoldedResonanceSpacingHistogramPlot,
                "nn_spacings",
                "Resonance Spacing Distribution",
            ),
            (
                UnfoldedComplexEnergyHistogramPlot,
                "complex_energies",
                "Complex Resonances",
            ),
            (
                UnfoldedResonanceFormFactorsPlot,
                "form_factors",
                "Resonance Form Factors",
            ),
        )
        for buffers, unfolding in unfolded_cases:
            for plot_cls, data_name, subject in plot_cases:
                expected_title = f"{unfolding} {subject}: {metadata}"
                with self.subTest(title=expected_title):
                    plot = plot_cls(
                        data=getattr(buffers, data_name),
                        context=simulation.manifest,
                    )
                    plot.set_derived_attributes()
                    self.assertEqual(plot.axes.title, expected_title)
                    self.assertNotIn("\n", plot.axes.title)

    def test_buffer_schema_filenames_metadata_and_iteration_order(self) -> None:
        expected_counts = {0: 10, 2: 32}
        for max_degree, expected_count in expected_counts.items():
            with self.subTest(max_degree=max_degree):
                simulation = ResonanceStatisticsSimulation(
                    compound=build_compound(max_degree=max_degree),
                    realizs=1,
                )

                coefficient_buffers = tuple(simulation.coefficient_buffers)
                averaged_buffers = tuple(simulation.ave_unfolded_buffers)
                variate_buffers = tuple(simulation.var_unfolded_buffers)

                self.assertEqual(len(tuple(simulation)), expected_count)
                self.assertEqual(
                    tuple(type(buffer) for buffer in simulation.raw_buffers),
                    (
                        ResonanceHistogram,
                        WidthHistogram,
                        ResonanceSpacingHistogram,
                        ComplexEnergyHistogram,
                        FormFactorsData,
                    ),
                )
                self.assertEqual(
                    tuple(buffer._file_name for buffer in simulation.raw_buffers),
                    (
                        "resonance_histogram",
                        "width_histogram",
                        "resonance_spacing_histogram",
                        "complex_energy_histogram",
                        "resonance_form_factors",
                    ),
                )
                self.assertEqual(
                    tuple(buffer.metadata["degree"] for buffer in coefficient_buffers),
                    tuple(range(1, max_degree + 1)),
                )
                self.assertTrue(
                    all(
                        isinstance(buffer, ResonanceCoefficientsHistogram)
                        for buffer in coefficient_buffers
                    )
                )
                self.assertEqual(
                    tuple(buffer.polynomial_degree for buffer in averaged_buffers),
                    tuple(range(1, max_degree + 1)),
                )
                self.assertEqual(
                    tuple(buffer.polynomial_degree for buffer in variate_buffers),
                    tuple(range(1, max_degree + 1)),
                )
                self.assertTrue(
                    all(
                        data.metadata.get("unfolding") == "raw"
                        for data in simulation.raw_buffers
                    )
                )
                self.assertTrue(
                    all(
                        data.metadata["unfolding"] == "weight"
                        for data in simulation.wgt_unfolded_buffers
                    )
                )
                for polynomial_degree, buffers in enumerate(averaged_buffers, start=1):
                    self.assertTrue(
                        all(
                            data.metadata
                            == {
                                "unfolding": "average",
                                "polynomial_degree": polynomial_degree,
                            }
                            for data in buffers
                        )
                    )
                for polynomial_degree, buffers in enumerate(variate_buffers, start=1):
                    self.assertTrue(
                        all(
                            data.metadata
                            == {
                                "unfolding": "variate",
                                "polynomial_degree": polynomial_degree,
                            }
                            for data in buffers
                        )
                    )

                if max_degree:
                    self.assertEqual(
                        tuple(data._file_name for data in averaged_buffers[0]),
                        (
                            "resonance_histogram_averaged_unfolded_degree_1",
                            "width_histogram_averaged_unfolded_degree_1",
                            "resonance_spacing_histogram_averaged_unfolded_degree_1",
                            "complex_energy_histogram_averaged_unfolded_degree_1",
                            "resonance_form_factors_averaged_unfolded_degree_1",
                        ),
                    )

                self.assertFalse(hasattr(simulation, "request"))
                self.assertFalse(hasattr(simulation, "result"))

    def test_poles_are_accumulated_and_finalized_in_raw_and_weight_buffers(
        self,
    ) -> None:
        simulation = ResonanceStatisticsSimulation(
            compound=build_compound(),
            realizs=1,
        )
        poles = np.array([-0.5 - 0.1j, 0.25 - 0.4j], dtype=np.complex128)
        resonance_centers = poles.real
        resonance_widths = -2 * poles.imag
        ensemble = simulation.compound.ensemble

        with patch.object(
            CompoundEnsemble,
            "resonances_stream",
            return_value=iter((poles,)),
        ):
            returned = simulation.execute()

        self.assertIsNone(returned)
        self.assertEqual(simulation.execution_state, ExecutionState.COMPLETE)

        raw = simulation.raw_buffers
        np.testing.assert_array_equal(
            raw.resonance_centers.counts,
            histogram_counts((resonance_centers,), bins=raw.resonance_centers.bins),
        )
        np.testing.assert_array_equal(
            raw.resonance_widths.counts,
            histogram_counts(
                (resonance_widths / ensemble.spectral_radius,),
                bins=raw.resonance_widths.bins,
            ),
        )
        np.testing.assert_array_equal(
            raw.nn_spacings.counts,
            histogram_counts((np.array([0.75]),), bins=raw.nn_spacings.bins),
        )
        np.testing.assert_array_equal(
            raw.complex_energies.counts,
            histogram2d_counts(
                resonance_centers / ensemble.spectral_radius,
                resonance_widths / ensemble.spectral_radius,
                histogram=raw.complex_energies,
            ),
        )
        self.assertAlmostEqual(float(np.sum(raw.complex_energies.histogram)), 1.0)

        raw_moment = np.mean(
            np.exp(-1j * np.outer(resonance_centers, raw.form_factors.times)),
            axis=0,
        )
        np.testing.assert_allclose(raw.form_factors.first_moment, raw_moment)
        np.testing.assert_allclose(
            raw.form_factors.form_factor,
            np.abs(raw_moment) ** 2,
        )
        np.testing.assert_allclose(
            raw.form_factors.single_realization_form_factor,
            np.abs(raw_moment) ** 2,
        )
        np.testing.assert_allclose(raw.form_factors.connected_form_factor, 0.0)

        weight_cdf = simulation.compound.resonance_density.weight_cdf
        unfolded_centers = unfold_values(
            resonance_centers,
            cdf=weight_cdf,
            dimension=ensemble.dimension,
        )
        unfolded_widths = unfold_widths(
            resonance_widths,
            centers=resonance_centers,
            cdf=weight_cdf,
            dimension=ensemble.dimension,
        )
        weight = simulation.wgt_unfolded_buffers
        np.testing.assert_array_equal(
            weight.resonance_centers.counts,
            histogram_counts((unfolded_centers,), bins=weight.resonance_centers.bins),
        )
        np.testing.assert_array_equal(
            weight.resonance_widths.counts,
            histogram_counts((unfolded_widths,), bins=weight.resonance_widths.bins),
        )
        np.testing.assert_array_equal(
            weight.complex_energies.counts,
            histogram2d_counts(
                resonance_centers / ensemble.spectral_radius,
                unfolded_widths,
                histogram=weight.complex_energies,
            ),
        )
        self.assertAlmostEqual(float(np.sum(weight.complex_energies.histogram)), 1.0)

        for data in simulation:
            if isinstance(data, (ResonanceHistogram, WidthHistogram)) and np.sum(
                data.counts
            ):
                np.testing.assert_allclose(
                    data.histogram,
                    data.counts / (np.sum(data.counts) * np.diff(data.bins)),
                )

    def test_seeded_sampling_calibration_and_unfolding_match_control(self) -> None:
        simulation = ResonanceStatisticsSimulation(
            compound=build_compound(max_degree=2, seed=314159),
            realizs=2,
        )
        control = build_compound(max_degree=2, seed=314159)
        initial_rng_state = deepcopy(simulation.compound.rng_state)

        control_factory = TruncatedPolynomialCDFFactory(
            density=control.resonance_density,
            degrees=(1, 2),
            density_name="resonance",
        )
        average_cdfs = control_factory.average_interpolators()
        poles_samples = tuple(control.resonances_stream(realizs=2))
        resonance_center_samples = tuple(poles.real for poles in poles_samples)
        resonance_width_samples = tuple(-2 * poles.imag for poles in poles_samples)

        simulation.execute()

        self.assertEqual(simulation.compound.rng_state, control.rng_state)
        self.assertEqual(simulation.manifest.rng["initial_state"], initial_rng_state)
        self.assertEqual(
            simulation.manifest.rng["final_state"],
            control.rng_state,
        )
        calibration = cast(
            dict[str, object], simulation.manifest.execution["calibration"]
        )
        self.assertEqual(calibration["timing"], "cached_during_execution")

        coefficient_samples = tuple(
            control.resonance_density.compute_variate_coeffs(resonance_centers)
            for resonance_centers in resonance_center_samples
        )

        raw_form_factors = simulation.raw_buffers.form_factors
        first_raw_moment = np.mean(
            np.exp(
                -1j
                * np.outer(
                    resonance_center_samples[0],
                    raw_form_factors.times,
                )
            ),
            axis=0,
        )
        np.testing.assert_allclose(
            raw_form_factors.single_realization_form_factor,
            np.abs(first_raw_moment) ** 2,
        )

        weight_form_factors = simulation.wgt_unfolded_buffers.form_factors
        first_weight_centers = unfold_values(
            resonance_center_samples[0],
            cdf=control.resonance_density.weight_cdf,
            dimension=control.ensemble.dimension,
        )
        first_weight_moment = np.mean(
            np.exp(-1j * np.outer(first_weight_centers, weight_form_factors.times)),
            axis=0,
        )
        np.testing.assert_allclose(
            weight_form_factors.single_realization_form_factor,
            np.abs(first_weight_moment) ** 2,
        )

        for cdf, buffers in zip(
            average_cdfs,
            simulation.ave_unfolded_buffers,
            strict=True,
        ):
            first_average_centers = unfold_values(
                resonance_center_samples[0],
                cdf=cdf,
                dimension=control.ensemble.dimension,
            )
            first_average_moment = np.mean(
                np.exp(
                    -1j
                    * np.outer(
                        first_average_centers,
                        buffers.form_factors.times,
                    )
                ),
                axis=0,
            )
            np.testing.assert_allclose(
                buffers.form_factors.single_realization_form_factor,
                np.abs(first_average_moment) ** 2,
            )

        first_variate_cdfs = control_factory.interpolators_from_coeffs(
            coefficient_samples[0]
        )
        for cdf, buffers in zip(
            first_variate_cdfs,
            simulation.var_unfolded_buffers,
            strict=True,
        ):
            first_variate_centers = unfold_values(
                resonance_center_samples[0],
                cdf=cdf,
                dimension=control.ensemble.dimension,
            )
            first_variate_moment = np.mean(
                np.exp(
                    -1j
                    * np.outer(
                        first_variate_centers,
                        buffers.form_factors.times,
                    )
                ),
                axis=0,
            )
            np.testing.assert_allclose(
                buffers.form_factors.single_realization_form_factor,
                np.abs(first_variate_moment) ** 2,
            )

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
                int(np.count_nonzero(expected_coefficients < histogram.bins[0])),
            )
            self.assertEqual(
                histogram.overflow,
                int(np.count_nonzero(expected_coefficients >= histogram.bins[-1])),
            )
            self.assertAlmostEqual(
                np.sum(histogram.histogram * np.diff(histogram.bins)),
                1.0 if np.sum(histogram.counts) else 0.0,
            )

        for cdf, buffers in zip(
            average_cdfs,
            simulation.ave_unfolded_buffers,
            strict=True,
        ):
            expected_centers = tuple(
                unfold_values(
                    resonance_centers,
                    cdf=cdf,
                    dimension=control.ensemble.dimension,
                )
                for resonance_centers in resonance_center_samples
            )
            expected_widths = tuple(
                unfold_widths(
                    resonance_widths,
                    centers=resonance_centers,
                    cdf=cdf,
                    dimension=control.ensemble.dimension,
                )
                for resonance_centers, resonance_widths in zip(
                    resonance_center_samples,
                    resonance_width_samples,
                    strict=True,
                )
            )
            np.testing.assert_array_equal(
                buffers.resonance_centers.counts,
                histogram_counts(expected_centers, bins=buffers.resonance_centers.bins),
            )
            np.testing.assert_array_equal(
                buffers.resonance_widths.counts,
                histogram_counts(expected_widths, bins=buffers.resonance_widths.bins),
            )

        variate_samples: list[list[np.ndarray]] = [[], []]
        for resonance_centers, resonance_widths, coefficients in zip(
            resonance_center_samples,
            resonance_width_samples,
            coefficient_samples,
            strict=True,
        ):
            variate_cdfs = control_factory.interpolators_from_coeffs(coefficients)
            for index, cdf in enumerate(variate_cdfs):
                variate_samples[index].append(
                    unfold_widths(
                        resonance_widths,
                        centers=resonance_centers,
                        cdf=cdf,
                        dimension=control.ensemble.dimension,
                    )
                )

        for buffers, unfolded_width_samples in zip(
            simulation.var_unfolded_buffers,
            variate_samples,
            strict=True,
        ):
            np.testing.assert_array_equal(
                buffers.resonance_widths.counts,
                histogram_counts(
                    tuple(unfolded_width_samples),
                    bins=buffers.resonance_widths.bins,
                ),
            )

    def test_failed_execution_is_terminal(self) -> None:
        simulation = ResonanceStatisticsSimulation(
            compound=build_compound(),
            realizs=1,
        )
        with (
            patch.object(
                ResonanceStatisticsSimulation,
                "_realize_resonance_statistics",
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

    def test_save_load_plot_dispatch_and_missing_data_validation(self) -> None:
        simulation = ResonanceStatisticsSimulation(
            compound=build_compound(max_degree=2, seed=314159),
            realizs=2,
        )
        simulation.execute()
        completed_rng_state = deepcopy(simulation.compound.rng_state)

        with tempfile.TemporaryDirectory() as temporary_directory:
            destination_directory = simulation.save(temporary_directory)
            restored_simulation = load_resonance_statistics_simulation(
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
                restored_simulation.compound.resonance_density.average_coeffs,
                simulation.compound.resonance_density.average_coeffs,
            )

            raw_resonance_plot = ResonanceHistogramPlot(
                data=restored_simulation.raw_buffers.resonance_centers,
                context=restored_simulation.manifest,
            )
            raw_resonance_plot.set_derived_attributes()
            self.assertIsNotNone(raw_resonance_plot._resonance_pdf)
            raw_pdf_peak = float(np.max(raw_resonance_plot._resonance_pdf, initial=0.0))
            raw_histogram_peak = float(
                np.max(raw_resonance_plot.data.histogram, initial=0.0)
            )
            self.assertGreater(
                raw_resonance_plot.ylim[1],
                max(raw_histogram_peak, raw_pdf_peak),
            )

            native_coefficient_plots = tuple(
                ResonanceCoefficientsHistogramPlot(
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

            plot_patchers = [
                patch.object(plot_cls, "plot", autospec=True)
                for plot_cls in RESONANCE_PLOT_CLASSES
            ]
            plot_mocks = [plot_patcher.start() for plot_patcher in plot_patchers]
            try:
                plot_resonance_statistics_simulation(directory=destination_directory)
            finally:
                for plot_patcher in reversed(plot_patchers):
                    plot_patcher.stop()

            expected_call_counts = (2, 1, 5, 1, 5, 1, 5, 5, 1, 1, 5)
            for plot_mock, expected_call_count in zip(
                plot_mocks,
                expected_call_counts,
                strict=True,
            ):
                self.assertEqual(plot_mock.call_count, expected_call_count)
            self.assertIsNotNone(plot_mocks[1].call_args.args[0].context)

            shared_coefficient_plots = tuple(
                call.args[0] for call in plot_mocks[0].call_args_list
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
            manifest = json.loads(original_manifest_text)
            manifest["execution"]["calibration"]["average_coefficients"] = []
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "invalid shape"):
                load_resonance_statistics_simulation(directory=destination_directory)
            manifest_path.write_text(original_manifest_text, encoding="utf-8")

            unexpected = ResonanceHistogram(
                _file_name="unexpected_resonance_histogram",
                support=(-1.0, 1.0),
            )
            unexpected.save(directory=destination_directory)
            with self.assertRaisesRegex(ValueError, "not part of the simulation"):
                load_resonance_statistics_simulation(directory=destination_directory)
            (destination_directory / unexpected.to_path).unlink()

            missing_data_path = (
                destination_directory / next(iter(restored_simulation)).to_path
            )
            missing_data_path.unlink()
            with self.assertRaisesRegex(ValueError, "Saved data .* is missing"):
                load_resonance_statistics_simulation(directory=destination_directory)
            with self.assertRaisesRegex(ValueError, "Saved data .* is missing"):
                restored_simulation.plot(destination_directory)

        self.assertEqual(simulation.compound.rng_state, completed_rng_state)

    def test_plot_configuration_uses_a_detached_compound(self) -> None:
        simulation = ResonanceStatisticsSimulation(
            compound=build_compound(seed=902),
            realizs=1,
        )
        simulation.execute()
        completed_rng_state = deepcopy(simulation.compound.rng_state)

        plot = ComplexEnergyHistogramPlot(
            data=simulation.raw_buffers.complex_energies,
            context=simulation.manifest,
        )
        plot.set_derived_attributes()

        self.assertIsNot(plot.compound, simulation.compound)
        self.assertEqual(simulation.compound.rng_state, completed_rng_state)

    def test_degree_zero_raw_resonance_plot_draws_polynomial_weight_pdf(
        self,
    ) -> None:
        simulation = ResonanceStatisticsSimulation(
            compound=build_compound(max_degree=0, seed=271),
            realizs=2,
        )
        simulation.execute()
        self.assertEqual(simulation.manifest.execution["calibration"], {})

        plot = ResonanceHistogramPlot(
            data=simulation.raw_buffers.resonance_centers,
            context=simulation.manifest,
        )
        plot.set_derived_attributes()

        self.assertIsNotNone(plot._resonance_centers)
        self.assertIsNotNone(plot._resonance_pdf)
        resonance_centers = cast(np.ndarray, plot._resonance_centers)
        resonance_pdf = cast(np.ndarray, plot._resonance_pdf)
        np.testing.assert_array_equal(
            resonance_centers,
            np.linspace(*plot.xlim, plot.num_points),
        )
        np.testing.assert_allclose(
            resonance_pdf,
            plot.compound.resonance_density.weight_pdf(resonance_centers),
        )
        self.assertGreater(plot.ylim[1], np.max(resonance_pdf))
        self.assertEqual(
            plot.legend.labels,
            ("simulation", "polynomial weight"),
        )
        self.assertEqual(len(plot.legend.handles), 2)

        plot.ax = MagicMock()
        with (
            patch.object(plot, "build_figure"),
            patch.object(plot, "draw_histogram"),
            patch.object(plot, "finish_plot"),
        ):
            plot.plot(Path("unused"))
        plot.ax.plot.assert_called_once()
        plotted_centers, plotted_pdf = plot.ax.plot.call_args.args
        np.testing.assert_array_equal(plotted_centers, resonance_centers)
        np.testing.assert_array_equal(plotted_pdf, resonance_pdf)

    def test_raw_resonance_plot_dynamically_contains_histogram_and_pdf(
        self,
    ) -> None:
        ensemble = GOE(
            num_majoranas=6,
            max_spectral_polynomial_degree=2,
            seed=211,
        )
        coupling = float(np.sqrt(ensemble.spectral_radius * 100.0))
        simulation = ResonanceStatisticsSimulation(
            compound=CompoundEnsemble(
                ensemble=ensemble,
                couplings=coupling,
            ),
            realizs=2,
        )
        simulation.execute()

        histogram = simulation.raw_buffers.resonance_centers
        original_bins = histogram.bins.copy()
        original_counts = histogram.counts.copy()
        original_density = histogram.histogram.copy()
        plot = ResonanceHistogramPlot(data=histogram, context=simulation.manifest)
        plot.set_derived_attributes()

        self.assertIsNotNone(plot._resonance_centers)
        self.assertIsNotNone(plot._resonance_pdf)
        resonance_centers = cast(np.ndarray, plot._resonance_centers)
        resonance_pdf = cast(np.ndarray, plot._resonance_pdf)
        np.testing.assert_array_equal(
            resonance_centers,
            np.linspace(*plot.xlim, plot.num_points),
        )
        np.testing.assert_allclose(
            resonance_pdf,
            plot.compound.resonance_density.variate_pdf(
                resonance_centers,
                coeffs=plot.calibration_coefficients("resonance"),
            ),
        )

        density_peak = max(
            float(np.max(histogram.histogram, initial=0.0)),
            float(np.max(resonance_pdf, initial=0.0)),
        )
        self.assertGreaterEqual(plot.ylim[1], 1.05 * density_peak)
        self.assertEqual(plot.ylim[0], 0.0)
        np.testing.assert_allclose(
            plot.xlim,
            (-1.2 * ensemble.spectral_radius, 1.2 * ensemble.spectral_radius),
        )
        np.testing.assert_allclose(
            plot.axes.xticks,
            (-ensemble.spectral_radius, 0.0, ensemble.spectral_radius),
        )
        self.assertEqual(
            plot.axes.xtick_labels,
            (r"$-1.0$", r"$0.0$", r"$+1.0$"),
        )

        self.assertTrue(np.all(np.diff(plot.axes.yticks) > 0.0))
        self.assertEqual(plot.axes.yticks[0], plot.ylim[0])
        self.assertEqual(plot.axes.yticks[-1], plot.ylim[1])
        np.testing.assert_allclose(
            plot.axes.yticks_minor,
            0.5 * (np.asarray(plot.axes.yticks[:-1]) + np.asarray(plot.axes.yticks[1:])),
        )
        self.assertEqual(len(plot.axes.ytick_labels), len(plot.axes.yticks))
        for tick, label in zip(
            plot.axes.yticks,
            plot.axes.ytick_labels,
            strict=True,
        ):
            label_value = float(label.removeprefix("$").removesuffix("$"))
            self.assertAlmostEqual(
                label_value,
                np.pi * ensemble.spectral_radius * tick,
            )

        derived_axis_configuration = (
            plot.xlim,
            plot.ylim,
            plot.axes.xticks,
            plot.axes.yticks,
            plot.axes.yticks_minor,
            plot.axes.ytick_labels,
        )
        plot.set_derived_attributes()
        self.assertEqual(
            derived_axis_configuration,
            (
                plot.xlim,
                plot.ylim,
                plot.axes.xticks,
                plot.axes.yticks,
                plot.axes.yticks_minor,
                plot.axes.ytick_labels,
            ),
        )

        plot.ax = MagicMock()
        with (
            patch.object(plot, "build_figure"),
            patch.object(plot, "draw_histogram"),
            patch.object(plot, "finish_plot"),
        ):
            plot.plot(Path("unused"))
        plot.ax.plot.assert_called_once()
        np.testing.assert_array_equal(histogram.bins, original_bins)
        np.testing.assert_array_equal(histogram.counts, original_counts)
        np.testing.assert_array_equal(histogram.histogram, original_density)

        empty_histogram = ResonanceHistogram.create_raw(
            resonance_density=simulation.compound.resonance_density,
        )
        pdf_dominant_plot = ResonanceHistogramPlot(
            data=empty_histogram,
            context=simulation.manifest,
        )
        pdf_dominant_plot.set_derived_attributes()
        self.assertIsNotNone(pdf_dominant_plot._resonance_pdf)
        self.assertGreater(
            pdf_dominant_plot.ylim[1],
            np.max(pdf_dominant_plot._resonance_pdf),
        )

        unfolded_plot = UnfoldedResonanceHistogramPlot(
            data=simulation.wgt_unfolded_buffers.resonance_centers,
            context=simulation.manifest,
        )
        unfolded_plot.set_derived_attributes()
        self.assertEqual(
            unfolded_plot.ylim,
            (0.0, 1.625 / ensemble.dimension),
        )

    def test_raw_resonance_plot_uses_histogram_and_empty_fallback_without_pdf(
        self,
    ) -> None:
        simulation = ResonanceStatisticsSimulation(
            compound=build_compound(max_degree=1, seed=433),
            realizs=1,
        )
        histogram = ResonanceHistogram.create_raw(
            resonance_density=simulation.compound.resonance_density,
        )
        histogram.histogram[histogram.num_bins // 2] = 1.0

        histogram_plot = ResonanceHistogramPlot(
            data=histogram,
            context=simulation.manifest,
        )
        histogram_plot.set_derived_attributes()
        self.assertIsNone(histogram_plot._resonance_pdf)
        self.assertGreaterEqual(histogram_plot.ylim[1], 1.05)
        self.assertEqual(len(histogram_plot.legend.handles), 1)
        self.assertEqual(len(histogram_plot.legend.labels), 1)

        empty_histogram = ResonanceHistogram.create_raw(
            resonance_density=simulation.compound.resonance_density,
        )
        fallback_plot = ResonanceHistogramPlot(
            data=empty_histogram,
            context=simulation.manifest,
        )
        fallback_plot.set_derived_attributes()
        expected_scale = np.pi * simulation.compound.ensemble.spectral_radius
        self.assertEqual(fallback_plot.ylim, (0.0, 2.6 / expected_scale))
        np.testing.assert_allclose(
            fallback_plot.axes.yticks,
            np.asarray((0.0, 1.0, 2.0)) / expected_scale,
        )

    def test_coefficient_plot_frames_central_mass_without_discarding_outlier(
        self,
    ) -> None:
        samples = np.append(np.linspace(-0.1, 0.1, 1_000), 10.0)
        bins = np.histogram_bin_edges(samples, bins="fd")
        bins[-1] = np.nextafter(bins[-1], np.inf)
        counts = np.histogram(samples, bins=bins)[0]
        histogram = ResonanceCoefficientsHistogram(
            metadata={"degree": 1, "unfolding": "raw"},
            _file_name="resonance_coeff_1_histogram",
            support=(float(bins[0]), float(bins[-1])),
            num_bins=len(counts),
            bins=bins,
            counts=counts,
            realizs=len(samples),
        )
        histogram.compute_histogram()
        original_bins = histogram.bins.copy()
        original_counts = histogram.counts.copy()
        original_density = histogram.histogram.copy()

        context = ResonanceStatisticsSimulation(
            compound=build_compound(seed=23),
            realizs=1,
        ).manifest
        plot = ResonanceCoefficientsHistogramPlot(data=histogram, context=context)
        plot.set_derived_attributes()

        self.assertEqual(np.sum(histogram.counts), len(samples))
        self.assertGreater(histogram.support[1], samples[-1])
        self.assertGreaterEqual(plot.xlim[1], np.max(samples[:-1]))
        self.assertLess(plot.xlim[1], samples[-1])
        self.assertEqual(plot.xlim[0], -plot.xlim[1])
        np.testing.assert_array_equal(histogram.bins, original_bins)
        np.testing.assert_array_equal(histogram.counts, original_counts)
        np.testing.assert_array_equal(histogram.histogram, original_density)

    def test_single_realization_and_empty_coefficient_histograms_are_valid(
        self,
    ) -> None:
        simulation = ResonanceStatisticsSimulation(
            compound=build_compound(max_degree=1, seed=77),
            realizs=1,
        )
        simulation.execute()

        histogram = next(iter(simulation.coefficient_buffers))
        self.assertEqual(histogram.realizs, 1)
        self.assertEqual(histogram.num_bins, 100)
        self.assertEqual(
            np.sum(histogram.counts) + histogram.underflow + histogram.overflow,
            1,
        )
        self.assertAlmostEqual(
            np.sum(histogram.histogram * np.diff(histogram.bins)),
            1.0,
        )

        empty_histogram = ResonanceCoefficientsHistogram.create(degree=1, dimension=2)
        empty_plot = ResonanceCoefficientsHistogramPlot(
            data=empty_histogram,
            context=simulation.manifest,
        )
        empty_plot.set_derived_attributes()

        self.assertEqual(empty_plot.xlim[0], -empty_plot.xlim[1])
        self.assertEqual(empty_plot.ylim[0], 0.0)
        self.assertGreater(empty_plot.ylim[1], 0.0)

    def test_fixed_support_coefficient_archive_remains_loadable(self) -> None:
        histogram = ResonanceCoefficientsHistogram.create(degree=1, dimension=2)
        histogram.add_histogram_contribution(np.array([-0.1, 0.0, 0.1]))
        histogram.compute_histogram()

        with tempfile.TemporaryDirectory() as temporary_directory:
            histogram.save(directory=temporary_directory)
            restored = ResonanceCoefficientsHistogram.load(
                Path(temporary_directory) / histogram.to_path
            )

        self.assertIsInstance(restored, ResonanceCoefficientsHistogram)
        self.assertEqual(restored.num_bins, histogram.num_bins)
        self.assertEqual(restored.support, histogram.support)
        np.testing.assert_array_equal(restored.bins, histogram.bins)
        np.testing.assert_array_equal(restored.counts, histogram.counts)
        np.testing.assert_array_equal(restored.histogram, histogram.histogram)

        context = ResonanceStatisticsSimulation(
            compound=build_compound(seed=41),
            realizs=1,
        ).manifest
        plot = ResonanceCoefficientsHistogramPlot(data=restored, context=context)
        plot.set_derived_attributes()
        self.assertEqual(plot.xlim[0], -plot.xlim[1])
        self.assertGreater(plot.ylim[1], np.max(restored.histogram))

    def test_run_helper_executes_saves_reloads_and_dispatches_plots(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            ExitStack() as plot_stack,
        ):
            plot_mocks = [
                plot_stack.enter_context(patch.object(plot_cls, "plot", autospec=True))
                for plot_cls in RESONANCE_PLOT_CLASSES
            ]
            simulation = run_resonance_statistics_simulation(
                compound=build_compound(seed=311),
                realizs=1,
                directory=temporary_directory,
            )

            self.assertEqual(simulation.execution_state, ExecutionState.COMPLETE)
            self.assertEqual(plot_mocks[0].call_count, 0)
            for plot_mock in plot_mocks[1:]:
                self.assertEqual(plot_mock.call_count, 1)

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
