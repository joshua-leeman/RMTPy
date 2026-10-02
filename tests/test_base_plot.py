import dataclasses
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from typing import cast, override
from unittest.mock import MagicMock, patch

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes

from rmtpy.ensembles import GOE, ManyBodyEnsemble
from rmtpy.simulations.base_data import Data
from rmtpy.simulations.base_plot import (
    LogDimensionTimeAxes,
    LogDimensionUnfoldedTimeAxes,
    Plot,
    PlotAxes,
    PlotLegend,
)
from rmtpy.simulations.resonance_statistics.resonance_form_factors import (
    ResonanceFormFactorsPlot,
    UnfoldedResonanceFormFactorsPlot,
)
from rmtpy.simulations.resonance_statistics.resonance_form_factors.resonance_form_factors_plot import (
    ResonanceFormFactorsAxes,
    UnfoldedResonanceFormFactorsAxes,
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
        legend_mock.get_title.return_value.set_linespacing.assert_called_once_with(1.5)
        legend_mock.get_title.return_value.set_color.assert_called_once_with("white")
        legend_text_mock.set_color.assert_called_once_with("white")

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
