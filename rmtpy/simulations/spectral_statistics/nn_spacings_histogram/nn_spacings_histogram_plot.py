import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import cast, override

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ....ensembles import ManyBodyEnsemble
from ...base_data import Data
from ...base_plot import CURVE_WIDTH, Plot, PlotAxes, PlotLegend


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class SpacingsHistogramAxes(PlotAxes):
    xticks: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0)  # units of mean spacing
    xticks_minor: tuple[float, ...] = (0.5, 1.5, 2.5, 3.5)
    xlabel: str = r"$s / \Delta$"
    xtick_labels: tuple[str, ...] = (
        r"$0.0$",
        r"$1.0$",
        r"$2.0$",
        r"$3.0$",
        r"$4.0$",
    )

    yticks: tuple[float, ...] = (0.5, 1.0)
    yticks_minor: tuple[float, ...] = (0.25, 0.75)
    ylabel: str = r"$P(s) \Delta$"
    ytick_labels: tuple[str, ...] = (
        r"$0.5$",
        r"$1.0$",
    )


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class SpacingsHistogramPlot(Plot):
    data: Data
    axes: PlotAxes = dataclasses.field(default_factory=SpacingsHistogramAxes)
    num_points: int = 1000

    xlim: tuple[float, float] = (0.0, 4.0)  # units of mean spacing
    ylim: tuple[float, float] = (0.0, 1.2)

    histogram_zorder: int = 1
    histogram_alpha: float = 0.5
    histogram_color: str = "Orange"
    histogram_legend: str = "simulation"

    surmise_zorder: int = 2
    surmise_width: float = CURVE_WIDTH
    surmise_alpha: float = 1.0
    surmise_color: str = "Black"
    surmise_legend: str = "surmise"

    legend_labels: tuple[str, str] = (histogram_legend, surmise_legend)
    legend_handles: tuple[Patch, Line2D] = (
        Patch(color=histogram_color, alpha=histogram_alpha),
        Line2D([0], [0], color=surmise_color, linewidth=surmise_width),
    )

    def set_derived_attributes(self) -> None:
        self.ensemble: ManyBodyEnsemble = self.store_manifest_arg(
            "ensemble", ManyBodyEnsemble
        )

        if self.ensemble.universality_class is not None:
            self.surmise_legend = f"{self.ensemble.universality_class} surmise"
            self.legend_labels = (self.histogram_legend, self.surmise_legend)

        self.legend: PlotLegend = PlotLegend(
            handles=self.legend_handles,
            labels=self.legend_labels,
            loc="upper right",
            bbox=(0.94, 0.95),
        )
        self.axes.title = "NNS Distribution: " + self.ensemble.to_latex

        mean_spacing = cast(float, self.data.metadata["global_mean_spacing"])

        self.scale_limits_and_ticks(
            x=lambda value: value * mean_spacing,
            y=lambda value: value / mean_spacing,
        )

    @override
    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        self.build_figure()

        self.draw_histogram(
            color=self.histogram_color,
            alpha=self.histogram_alpha,
            zorder=self.histogram_zorder,
        )

        spacings = np.linspace(*self.xlim, self.num_points)
        mean_spacing = cast(float, self.data.metadata["global_mean_spacing"])

        surmise = self.ensemble.wigner_surmise(spacings / mean_spacing)
        surmise /= mean_spacing

        plot = cast(Callable[..., object], self.ax.plot)
        _ = plot(
            spacings,
            surmise,
            color=self.surmise_color,
            linewidth=self.surmise_width,
            alpha=self.surmise_alpha,
            zorder=self.surmise_zorder,
        )

        self.finish_plot(path=path)


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class UnfoldedSpacingsHistogramAxes(PlotAxes):
    xticks: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0)
    xticks_minor: tuple[float, ...] = (0.5, 1.5, 2.5, 3.5)
    xlabel: str = r"$\sigma$"
    xtick_labels: tuple[str, ...] = (
        r"$0.0$",
        r"$1.0$",
        r"$2.0$",
        r"$3.0$",
        r"$4.0$",
    )

    yticks: tuple[float, ...] = (0.5, 1.0)
    yticks_minor: tuple[float, ...] = (0.25, 0.75)
    ylabel: str = r"$P(\sigma)$"
    ytick_labels: tuple[str, ...] = (
        r"$0.5$",
        r"$1.0$",
    )


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class UnfoldedSpacingsHistogramPlot(Plot):
    data: Data
    axes: PlotAxes = dataclasses.field(default_factory=UnfoldedSpacingsHistogramAxes)
    num_points: int = 1000

    xlim: tuple[float, float] = (0.0, 4.0)
    ylim: tuple[float, float] = (0.0, 1.2)

    histogram_zorder: int = 1
    histogram_alpha: float = 0.5
    histogram_color: str = "Orange"
    histogram_legend: str = "simulation"

    surmise_zorder: int = 2
    surmise_width: float = CURVE_WIDTH
    surmise_alpha: float = 1.0
    surmise_color: str = "Black"
    surmise_legend: str = "surmise"

    legend_labels: tuple[str, str] = (histogram_legend, surmise_legend)
    legend_handles: tuple[Patch, Line2D] = (
        Patch(color=histogram_color, alpha=histogram_alpha),
        Line2D([0], [0], color=surmise_color, linewidth=surmise_width),
    )

    def set_derived_attributes(self) -> None:
        self.ensemble: ManyBodyEnsemble = self.store_manifest_arg(
            "ensemble", ManyBodyEnsemble
        )

        if self.ensemble.universality_class is not None:
            self.surmise_legend = f"{self.ensemble.universality_class} surmise"
            self.legend_labels = (self.histogram_legend, self.surmise_legend)

        self.legend: PlotLegend = PlotLegend(
            handles=self.legend_handles,
            labels=self.legend_labels,
            loc="upper right",
            bbox=(0.94, 0.95),
        )

        unfolding_type = cast(str, self.data.metadata["unfolding"])
        unfolding_label = (
            "Average" if unfolding_type == "averaged" else unfolding_type.capitalize()
        )
        title = f"{unfolding_label}-unfolded"
        if unfolding_type != "weight":
            unfolding_degree = self.data.metadata["polynomial_degree"]
            title += f" (deg = ${unfolding_degree}$)"
        self.axes.title = f"{title} NNS Distribution: {self.ensemble.to_latex}"

    @override
    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        self.build_figure()

        self.draw_histogram(
            color=self.histogram_color,
            alpha=self.histogram_alpha,
            zorder=self.histogram_zorder,
        )

        spacings = np.linspace(0, self.xlim[1], self.num_points)
        surmise = self.ensemble.wigner_surmise(spacings)

        plot = cast(Callable[..., object], self.ax.plot)
        _ = plot(
            spacings,
            surmise,
            color=self.surmise_color,
            linewidth=self.surmise_width,
            alpha=self.surmise_alpha,
            zorder=self.surmise_zorder,
        )

        self.finish_plot(path=path)
