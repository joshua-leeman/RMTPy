import dataclasses
import shutil
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from typing import cast, override
from unittest.mock import MagicMock, patch

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator

from rmtpy.ensembles import GOE, ManyBodyEnsemble
from rmtpy.simulations.base_data import Data
from rmtpy.simulations.base_plot import (
    LogDimensionTimeAxes,
    LogDimensionUnfoldedTimeAxes,
    Plot,
    PlotAxes,
    PlotLegend,
)
from rmtpy.simulations.partial_widths_statistics.partial_width_histogram.partial_width_histogram_plot import (
    PartialWidthHistogramAxes,
)
from rmtpy.simulations.partial_widths_statistics.total_width_histogram.total_width_histogram_plot import (
    TotalWidthHistogramAxes,
)
from rmtpy.simulations.resonance_statistics.resonance_form_factors import (
    ResonanceFormFactorsPlot,
    UnfoldedResonanceFormFactorsPlot,
)
from rmtpy.simulations.resonance_statistics.resonance_form_factors.resonance_form_factors_plot import (
    ResonanceFormFactorsAxes,
    UnfoldedResonanceFormFactorsAxes,
)
from rmtpy.simulations.resonance_statistics.width_histogram.width_histogram_plot import (
    UnfoldedWidthHistogramAxes,
    WidthHistogramAxes,
)
from rmtpy.simulations.spectral_statistics import SpectralStatisticsSimulation
from rmtpy.simulations.spectral_statistics.spectral_form_factors import (
    FormFactorsPlot,
    UnfoldedFormFactorsPlot,
)
from rmtpy.simulations.spectral_statistics.spectral_form_factors.spectral_form_factors_plot import (
    FormFactorsAxes,
    UnfoldedFormFactorsAxes,
)
from rmtpy.simulations.statistics import LOG_D_TIME_SUPPORT, LOG_D_UNFOLDED_TIME_SUPPORT
from rmtpy.simulations.time_delay_statistics.time_delay_histogram import (
    TimeDelayHistogramPlot,
    UnfoldedTimeDelayHistogramPlot,
)
from rmtpy.simulations.time_delay_statistics.time_delay_histogram.time_delay_histogram_plot import (
    TimeDelayHistogramAxes,
    UnfoldedTimeDelayHistogramAxes,
)


class _CompleteFigure:
    def savefig(self, path: str | Path, **_arguments: object) -> None:
        Path(path).write_bytes(b"complete png")


class _FailingFigure:
    def savefig(self, path: str | Path, **_arguments: object) -> None:
        Path(path).write_bytes(b"partial png")
        raise OSError("expected plot failure")


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class _ConcretePlot(Plot):
    @override
    def plot(self, path: str | Path) -> None:
        self.build_figure()
        self.finish_plot(path)


def build_simulation(*, seed: int = 123) -> SpectralStatisticsSimulation:
    return SpectralStatisticsSimulation(
        ensemble=GOE(num_majoranas=4, seed=seed),
        realizs=1,
    )


def default_dataclass_field(plot_cls: type[Plot], *, name: str) -> object:
    for field in dataclasses.fields(plot_cls):
        if field.name == name:
            return field.default

    raise AssertionError(f"{plot_cls.__name__} has no `{name}` field.")


class BasePlotTests(unittest.TestCase):
    def test_manifest_arguments_are_detached_cached_and_nonmutating(self) -> None:
        simulation = build_simulation(seed=902)
        configuration_before = deepcopy(simulation.manifest.configuration)
        rng_state_before = deepcopy(simulation.ensemble.rng_state)
        plot = _ConcretePlot(
            data=Data(_file_name="example"),
            context=simulation.manifest,
        )

        first_ensemble = plot.store_manifest_arg("ensemble", ManyBodyEnsemble)
        second_ensemble = plot.store_manifest_arg("ensemble", ManyBodyEnsemble)

        self.assertIs(first_ensemble, second_ensemble)
        self.assertIsNot(first_ensemble, simulation.ensemble)
        self.assertEqual(simulation.manifest.configuration, configuration_before)
        self.assertEqual(simulation.ensemble.rng_state, rng_state_before)

        with self.assertRaisesRegex(ValueError, "parameter not found"):
            plot.store_manifest_arg("compound", ManyBodyEnsemble)

    def test_calibration_coefficients_are_selected_and_validated(self) -> None:
        simulation = build_simulation()
        simulation.manifest.execution["calibration"] = {
            "density": "spectral",
            "average_coefficients": [1.0, 0.25],
        }
        plot = _ConcretePlot(
            data=Data(_file_name="example"),
            context=simulation.manifest,
        )

        coefficients = plot.calibration_coefficients("spectral")
        self.assertIsNotNone(coefficients)
        coefficients = cast(np.ndarray[tuple[int], np.dtype[np.floating]], coefficients)
        self.assertEqual(coefficients.tolist(), [1.0, 0.25])
        self.assertIsNone(plot.calibration_coefficients("resonance"))

        simulation.manifest.execution["calibration"] = {
            "density": "spectral",
            "average_coefficients": [[1.0]],
        }
        with self.assertRaisesRegex(ValueError, "malformed"):
            plot.calibration_coefficients("spectral")

        simulation.manifest.execution["calibration"] = {
            "density": "spectral",
        }
        with self.assertRaisesRegex(ValueError, "missing"):
            plot.calibration_coefficients("spectral")

    def test_axes_and_legend_configuration_apply_complete_style(self) -> None:
        axes_mock = MagicMock()
        axes_mock.get_xscale.return_value = "linear"
        axes_mock.get_yscale.return_value = "linear"
        left_spine = MagicMock()
        right_spine = MagicMock()
        axes_mock.spines = {"left": left_spine, "right": right_spine}
        axes_configuration = PlotAxes(
            axes_width=1.5,
            xlabel="horizontal",
            ylabel="vertical",
            xticks=(0.0, 1.0),
            yticks=(2.0, 3.0),
            xticks_minor=(0.5,),
            yticks_minor=(2.5,),
            xtick_labels=("zero", "one"),
            ytick_labels=("two", "three"),
        )

        axes_configuration.configure(axes=axes_mock)

        left_spine.set_linewidth.assert_called_once_with(1.5)
        right_spine.set_linewidth.assert_called_once_with(1.5)
        axes_mock.set_xlabel.assert_called_once_with("horizontal", fontsize=12)
        axes_mock.set_ylabel.assert_called_once_with("vertical", fontsize=12)
        axes_mock.set_xticks.assert_any_call((0.0, 1.0))
        axes_mock.set_xticks.assert_any_call((0.5,), minor=True)
        axes_mock.set_yticks.assert_any_call((2.0, 3.0))
        axes_mock.set_yticks.assert_any_call((2.5,), minor=True)
        axes_mock.set_xticklabels.assert_called_once_with(
            ("zero", "one"),
            fontsize=10,
        )
        axes_mock.set_yticklabels.assert_called_once_with(
            ("two", "three"),
            fontsize=10,
        )

        legend_axes_mock = MagicMock()
        legend_mock = MagicMock()
        legend_text_mock = MagicMock()
        legend_mock.get_texts.return_value = (legend_text_mock,)
        legend_axes_mock.legend.return_value = legend_mock
        handle = cast(Artist, MagicMock())
        legend_configuration = PlotLegend(
            handles=(handle,),
            labels=("curve",),
            title="reference",
            on_black_background=True,
            loc="upper right",
            bbox=(1.0, 1.0),
        )

        legend_configuration.configure(ax=cast(Axes, legend_axes_mock))

        legend_axes_mock.legend.assert_called_once_with(
            handles=(handle,),
            labels=("curve",),
            title="reference",
            loc="upper right",
            bbox_to_anchor=(1.0, 1.0),
            frameon=False,
            fontsize=10,
            title_fontsize=10,
            alignment="left",
        )
        legend_mock.get_title.return_value.set_linespacing.assert_not_called()
        legend_text_mock.set_linespacing.assert_not_called()
        legend_mock.get_title.return_value.set_color.assert_called_once_with("white")
        legend_text_mock.set_color.assert_called_once_with("white")

    def test_legend_formats_only_multiline_titles_for_tex(self) -> None:
        legend = PlotLegend(title_linegap=2.0)

        for title, expected in (
            ("single line", "single line"),
            (
                "first line\nsecond line",
                r"\shortstack[l]{first line\\[2pt]second line}",
            ),
            (
                "first line\nsecond line\nthird line",
                (
                    r"\shortstack[l]{first line\\[2pt]second line"
                    r"\\[2pt]third line}"
                ),
            ),
        ):
            with self.subTest(title=title):
                legend.title = title
                self.assertEqual(legend._formatted_title(usetex=True), expected)

        legend.title = "first line\nsecond line"
        legend.title_linegap = 3.5
        self.assertEqual(
            legend._formatted_title(usetex=True),
            r"\shortstack[l]{first line\\[3.5pt]second line}",
        )

    def test_non_tex_multiline_legend_title_uses_equivalent_line_spacing(
        self,
    ) -> None:
        axes_mock = MagicMock()
        legend_mock = MagicMock()
        axes_mock.legend.return_value = legend_mock
        handle = cast(Artist, MagicMock())
        legend = PlotLegend(
            handles=(handle,),
            labels=("curve",),
            title="first line\nsecond line",
            title_fontsize=10,
            title_linegap=2.0,
        )

        with matplotlib.rc_context({"text.usetex": False}):
            legend.configure(ax=cast(Axes, axes_mock))

        axes_mock.legend.assert_called_once_with(
            handles=(handle,),
            labels=("curve",),
            title="first line\nsecond line",
            loc="best",
            bbox_to_anchor=(),
            frameon=False,
            fontsize=10,
            title_fontsize=10,
            alignment="left",
        )
        legend_mock.get_title.return_value.set_linespacing.assert_called_once_with(1.2)

    @unittest.skipUnless(
        shutil.which("latex") and shutil.which("dvipng"),
        "LaTeX rendering tools are unavailable",
    )
    def test_tex_linegap_increases_only_multiline_legend_title_height(self) -> None:
        title = "GOE($N_m = 14$)\n$N_f = 2$, $a = 1$"
        handle = Line2D([0], [0])

        with matplotlib.rc_context({"text.usetex": True}):
            control_figure, control_axes = plt.subplots()
            spaced_figure, spaced_axes = plt.subplots()
            try:
                control_legend = control_axes.legend(
                    handles=(handle,),
                    labels=("simulation",),
                    title=title,
                )
                PlotLegend(
                    handles=(handle,),
                    labels=("simulation",),
                    title=title,
                    title_linegap=2.0,
                    bbox=(1.0, 1.0),
                ).configure(spaced_axes)
                spaced_legend = spaced_axes.get_legend()
                self.assertIsNotNone(spaced_legend)

                control_figure.canvas.draw()
                spaced_figure.canvas.draw()
                control_renderer = control_figure.canvas.get_renderer()
                spaced_renderer = spaced_figure.canvas.get_renderer()
                control_title_box = control_legend.get_title().get_window_extent(
                    control_renderer
                )
                spaced_title_box = spaced_legend.get_title().get_window_extent(
                    spaced_renderer
                )
                control_entry_box = control_legend.get_texts()[0].get_window_extent(
                    control_renderer
                )
                spaced_entry_box = spaced_legend.get_texts()[0].get_window_extent(
                    spaced_renderer
                )

                self.assertGreater(spaced_title_box.height, control_title_box.height)
                self.assertAlmostEqual(
                    spaced_title_box.y0 - spaced_entry_box.y1,
                    control_title_box.y0 - control_entry_box.y1,
                )
                self.assertEqual(spaced_legend.get_texts()[0].get_text(), "simulation")
            finally:
                plt.close(control_figure)
                plt.close(spaced_figure)

    def test_all_log_log_axes_use_single_unlabeled_half_length_minor_ticks(
        self,
    ) -> None:
        axes_configurations = (
            (WidthHistogramAxes, 10),
            (UnfoldedWidthHistogramAxes, 10),
            (PartialWidthHistogramAxes, 10),
            (TotalWidthHistogramAxes, 10),
            (FormFactorsAxes, 16),
            (UnfoldedFormFactorsAxes, 16),
            (ResonanceFormFactorsAxes, 16),
            (UnfoldedResonanceFormFactorsAxes, 16),
        )

        for axes_cls, base in axes_configurations:
            with self.subTest(axes_cls=axes_cls):
                axes_configuration = axes_cls()
                axes_configuration.xticks = tuple(
                    base**value for value in axes_configuration.xticks
                )
                axes_configuration.yticks = tuple(
                    base**value for value in axes_configuration.yticks
                )

                figure, axes = plt.subplots()
                try:
                    axes.set_xscale("log", base=base)
                    axes.set_yscale("log", base=base)
                    axes_configuration.configure(axes)

                    expected_x_minor_ticks = tuple(
                        np.sqrt(left_tick * right_tick)
                        for left_tick, right_tick in zip(
                            axes_configuration.xticks,
                            axes_configuration.xticks[1:],
                            strict=False,
                        )
                    )
                    expected_y_minor_ticks = tuple(
                        np.sqrt(left_tick * right_tick)
                        for left_tick, right_tick in zip(
                            axes_configuration.yticks,
                            axes_configuration.yticks[1:],
                            strict=False,
                        )
                    )
                    np.testing.assert_allclose(
                        axes.xaxis.get_minorticklocs(),
                        expected_x_minor_ticks,
                    )
                    np.testing.assert_allclose(
                        axes.yaxis.get_minorticklocs(),
                        expected_y_minor_ticks,
                    )

                    x_minor_ticks = axes.xaxis.get_minor_ticks()
                    y_minor_ticks = axes.yaxis.get_minor_ticks()
                    self.assertEqual(len(x_minor_ticks), len(expected_x_minor_ticks))
                    self.assertEqual(len(y_minor_ticks), len(expected_y_minor_ticks))
                    for major_tick in (
                        *axes.xaxis.get_major_ticks(),
                        *axes.yaxis.get_major_ticks(),
                    ):
                        self.assertEqual(
                            major_tick.tick1line.get_markersize(),
                            axes_configuration.tick_length,
                        )
                        self.assertEqual(
                            major_tick.tick2line.get_markersize(),
                            axes_configuration.tick_length,
                        )
                    for minor_tick in (*x_minor_ticks, *y_minor_ticks):
                        self.assertEqual(
                            minor_tick.tick1line.get_markersize(),
                            axes_configuration.tick_length / 2,
                        )
                        self.assertEqual(
                            minor_tick.tick2line.get_markersize(),
                            axes_configuration.tick_length / 2,
                        )
                        self.assertFalse(minor_tick.label1.get_visible())
                        self.assertFalse(minor_tick.label2.get_visible())
                finally:
                    plt.close(figure)

    def test_semilog_axes_retain_automatic_full_length_minor_ticks(self) -> None:
        axes_configuration = PlotAxes(
            xticks=(1.0, 10.0, 100.0),
            yticks=(0.0, 1.0, 2.0),
        )
        figure, axes = plt.subplots()
        try:
            axes.set_xscale("log", base=10)
            axes.set_yscale("linear")
            axes.set_xlim(1.0, 100.0)
            axes_configuration.configure(axes)

            self.assertIsInstance(axes.xaxis.get_minor_locator(), LogLocator)
            self.assertTrue(
                all(
                    minor_tick.tick1line.get_markersize()
                    == axes_configuration.tick_length
                    for minor_tick in axes.xaxis.get_minor_ticks()
                )
            )
        finally:
            plt.close(figure)

    def test_plot_output_is_atomic_and_never_overwritten(self) -> None:
        simulation = build_simulation()
        plot = _ConcretePlot(
            data=Data(_file_name="example"),
            context=simulation.manifest,
        )
        object.__setattr__(plot, "fig", cast(object, _CompleteFigure()))
        object.__setattr__(plot, "ax", cast(object, object()))

        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            patch.object(PlotAxes, "configure"),
            patch.object(PlotLegend, "configure"),
        ):
            destination_directory = Path(temporary_directory)
            plot.finish_plot(destination_directory)

            destination_path = destination_directory / "example_plot.png"
            self.assertEqual(destination_path.read_bytes(), b"complete png")
            self.assertEqual(
                tuple(destination_directory.glob("*.partial.png")),
                (),
            )

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                plot.finish_plot(destination_directory)
            self.assertEqual(destination_path.read_bytes(), b"complete png")

    def test_failed_plot_output_leaves_no_partial_or_destination_file(self) -> None:
        simulation = build_simulation()
        plot = _ConcretePlot(
            data=Data(_file_name="example"),
            context=simulation.manifest,
        )
        object.__setattr__(plot, "fig", cast(object, _FailingFigure()))
        object.__setattr__(plot, "ax", cast(object, object()))

        with tempfile.TemporaryDirectory() as temporary_directory:
            with (
                patch.object(PlotAxes, "configure"),
                patch.object(PlotLegend, "configure"),
                self.assertRaisesRegex(OSError, "expected plot failure"),
            ):
                plot.finish_plot(temporary_directory)

            self.assertEqual(tuple(Path(temporary_directory).iterdir()), ())

    def test_time_based_outputs_share_axes_and_support_conventions(self) -> None:
        raw_axes_classes = (
            FormFactorsAxes,
            ResonanceFormFactorsAxes,
            TimeDelayHistogramAxes,
        )
        for axes_cls in raw_axes_classes:
            with self.subTest(axes_cls=axes_cls):
                axes = axes_cls()
                reference_axes = LogDimensionTimeAxes()
                self.assertIsInstance(axes, LogDimensionTimeAxes)
                self.assertEqual(axes.xticks, reference_axes.xticks)
                self.assertEqual(axes.xtick_labels, reference_axes.xtick_labels)
                self.assertEqual(axes.xlabel, reference_axes.xlabel)

        unfolded_axes_classes = (
            UnfoldedFormFactorsAxes,
            UnfoldedResonanceFormFactorsAxes,
            UnfoldedTimeDelayHistogramAxes,
        )
        for axes_cls in unfolded_axes_classes:
            with self.subTest(axes_cls=axes_cls):
                axes = axes_cls()
                reference_axes = LogDimensionUnfoldedTimeAxes()
                self.assertIsInstance(axes, LogDimensionUnfoldedTimeAxes)
                self.assertEqual(axes.xticks, reference_axes.xticks)
                self.assertEqual(axes.xtick_labels, reference_axes.xtick_labels)
                self.assertEqual(axes.xlabel, reference_axes.xlabel)

        for plot_cls in (
            FormFactorsPlot,
            ResonanceFormFactorsPlot,
            TimeDelayHistogramPlot,
        ):
            with self.subTest(plot_cls=plot_cls):
                self.assertEqual(
                    default_dataclass_field(plot_cls, name="xlim"),
                    LOG_D_TIME_SUPPORT,
                )

        for plot_cls in (
            UnfoldedFormFactorsPlot,
            UnfoldedResonanceFormFactorsPlot,
            UnfoldedTimeDelayHistogramPlot,
        ):
            with self.subTest(plot_cls=plot_cls):
                self.assertEqual(
                    default_dataclass_field(plot_cls, name="xlim"),
                    LOG_D_UNFOLDED_TIME_SUPPORT,
                )


if __name__ == "__main__":
    unittest.main()
