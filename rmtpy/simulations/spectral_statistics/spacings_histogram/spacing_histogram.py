import dataclasses
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from rmtpy.ensembles import ManyBodyEnsemble

from ...histogram import Histogram
from ...plot import Plot, PlotAxes, PlotLegend


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class SpacingsHistogramLegend(PlotLegend):
    loc: str = "upper right"
    bbox: tuple[float, float] = (0.94, 0.95)


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class SpacingsHistogramAxes(PlotAxes):
    xticks: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0)  # units of mean spacing
    xticks_minor: tuple[float, ...] = (0.5, 1.5, 2.5, 3.5)
    xlabel: str = r"$\Delta E$"
    xtick_labels: tuple[str, ...] = (
        r"$0$",
        r"$d$",
        r"$2d$",
        r"$3d$",
        r"$4d$",
    )

    yticks: tuple[float, ...] = (0.5, 1.0)
    yticks_minor: tuple[float, ...] = (0.25, 0.75)
    ylabel: str = r"$\ensavg{f(\Delta E)}$"
    ytick_labels: tuple[str, ...] = (
        r"$\frac{1}{2}d^{-1}$",
        r"$d^{-1}$",
    )


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class SpacingsHistogramPlot(Plot):
    data: Histogram
    axes: SpacingsHistogramAxes = dataclasses.field(
        default_factory=SpacingsHistogramAxes
    )
    num_points: int = 1000

    xlim: tuple[float, float] = (0.0, 4.0)  # units of mean spacing
    ylim: tuple[float, float] = (0.0, 1.2)

    histogram_zorder: int = 1
    histogram_alpha: float = 0.5
    histogram_color: str = "Orange"
    histogram_legend: str = "simulation"

    surmise_zorder: int = 2
    surmise_width: float = 2.0
    surmise_alpha: float = 1.0
    surmise_color: str = "Black"
    surmise_legend: str = "surmise"

    legend_labels: tuple[str, str] = (histogram_legend, surmise_legend)
    legend_handles: tuple[Patch, Line2D] = (
        Patch(color=histogram_color, alpha=histogram_alpha),
        Line2D([0], [0], color=surmise_color, linewidth=surmise_width),
    )

    def set_derived_attributes(self) -> None:
        self.ensemble: ManyBodyEnsemble = self.structure_simulation_arg(
            "ensemble", ManyBodyEnsemble
        )

        if self.ensemble.universality_class is not None:
            self.surmise_legend = f"{self.ensemble.universality_class} surmise"
            self.legend_labels = (self.histogram_legend, self.surmise_legend)

        self.legend: SpacingsHistogramLegend = SpacingsHistogramLegend(
            handles=self.legend_handles, labels=self.legend_labels
        )
        if self.legend.title is None:
            self.legend.title = self.ensemble.to_latex

        mean_spacing: float = self.data.metadata["global_mean_spacing"]

        self.scale_limits_and_ticks(
            x=lambda value: value * mean_spacing,
            y=lambda value: value / mean_spacing,
        )

    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        self.create_figure()

        self.draw_histogram(
            color=self.histogram_color,
            alpha=self.histogram_alpha,
            zorder=self.histogram_zorder,
        )

        spacings: np.ndarray = np.linspace(*self.xlim, self.num_points)
        mean_spacing: float = self.data.metadata["global_mean_spacing"]
        surmise: np.ndarray = self.ensemble.wigner_surmise(spacings / mean_spacing)
        surmise /= mean_spacing

        self.ax.plot(
            spacings,
            surmise,
            color=self.surmise_color,
            linewidth=self.surmise_width,
            alpha=self.surmise_alpha,
            zorder=self.surmise_zorder,
        )

        self.finish_plot(path=path)


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class UnfoldedSpacingsHistogramLegend(PlotLegend):
    loc: str = "upper right"
    bbox: tuple[float, float] = (0.94, 0.95)


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class UnfoldedSpacingsHistogramAxes(PlotAxes):
    xticks: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0)
    xticks_minor: tuple[float, ...] = (0.5, 1.5, 2.5, 3.5)
    xlabel: str = r"$s$"
    xtick_labels: tuple[str, ...] = (
        r"$0.0$",
        r"$1.0$",
        r"$2.0$",
        r"$3.0$",
        r"$4.0$",
    )

    yticks: tuple[float, ...] = (0.5, 1.0)
    yticks_minor: tuple[float, ...] = (0.25, 0.75)
    ylabel: str = r"$\ensavg{f(s)}$"
    ytick_labels: tuple[str, ...] = (
        r"$0.5$",
        r"$1.0$",
    )


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class UnfoldedSpacingsHistogramPlot(Plot):
    data: Histogram
    axes: UnfoldedSpacingsHistogramAxes = dataclasses.field(
        default_factory=UnfoldedSpacingsHistogramAxes
    )
    num_points: int = 1000

    xlim: tuple[float, float] = (0.0, 4.0)
    ylim: tuple[float, float] = (0.0, 1.2)

    histogram_zorder: int = 1
    histogram_alpha: float = 0.5
    histogram_color: str = "Orange"
    histogram_legend: str = "simulation"

    surmise_zorder: int = 2
    surmise_width: float = 2.0
    surmise_alpha: float = 1.0
    surmise_color: str = "Black"
    surmise_legend: str = "surmise"

    legend_labels: tuple[str, str] = (histogram_legend, surmise_legend)
    legend_handles: tuple[Patch, Line2D] = (
        Patch(color=histogram_color, alpha=histogram_alpha),
        Line2D([0], [0], color=surmise_color, linewidth=surmise_width),
    )

    def set_derived_attributes(self) -> None:
        self.ensemble: ManyBodyEnsemble = self.structure_simulation_arg(
            "ensemble", ManyBodyEnsemble
        )

        if self.ensemble.universality_class is not None:
            self.surmise_legend = f"{self.ensemble.universality_class} surmise"
            self.legend_labels = (self.histogram_legend, self.surmise_legend)

        self.legend: UnfoldedSpacingsHistogramLegend = UnfoldedSpacingsHistogramLegend(
            handles=self.legend_handles, labels=self.legend_labels
        )
        if self.legend.title is None:
            self.legend.title = self.ensemble.to_latex + "\nunfolded"

    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        self.create_figure()

        self.draw_histogram(
            color=self.histogram_color,
            alpha=self.histogram_alpha,
            zorder=self.histogram_zorder,
        )

        spacings: np.ndarray = np.linspace(0, self.xlim[1], self.num_points)
        surmise: np.ndarray = self.ensemble.wigner_surmise(spacings)

        self.ax.plot(
            spacings,
            surmise,
            color=self.surmise_color,
            linewidth=self.surmise_width,
            alpha=self.surmise_alpha,
            zorder=self.surmise_zorder,
        )

        self.finish_plot(path=path)
