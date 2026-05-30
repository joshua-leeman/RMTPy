import dataclasses
from pathlib import Path

from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter

import rmtpy.density
from rmtpy.compounds import Compound

from ...histogram import Histogram
from ...plot import Plot, PlotAxes, PlotLegend


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class PartialWidthHistogramLegend(PlotLegend):
    loc: str = "upper right"
    bbox: tuple[float, float] = (0.94, 0.95)


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class PartialWidthHistogramAxes(PlotAxes):
    xticks: tuple[float, ...] = tuple(range(-2, 2))  # log scale base 10
    xticks_minor: tuple[float, ...] = tuple()
    xlabel: str = r"$y = \lvert \gamma \rvert^{2} / \ensavg{ \lvert \gamma \rvert^{2}}$"
    xtick_labels: tuple[str, ...] = (
        r"$10^{-2}$",
        r"$10^{-1}$",
        r"$10^{0}$",
        r"$10^{1}$",
    )

    yticks: tuple[float, ...] = tuple(range(-3, 2))  # log scale base 10
    yticks_minor: tuple[float, ...] = tuple()
    ylabel: str = r"$P(y)$"
    ytick_labels: tuple[str, ...] = (
        r"$10^{-3}$",
        r"$10^{-2}$",
        r"$10^{-1}$",
        r"$10^{0}$",
        r"$10^{1}$",
    )


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class PartialWidthHistogramPlot(Plot):
    data: Histogram
    axes: PartialWidthHistogramAxes = dataclasses.field(
        default_factory=PartialWidthHistogramAxes
    )

    xlim: tuple[float, float] = (-2.4, 1.4)  # log scale base 10
    ylim: tuple[float, float] = (-3.2, 1.2)  # log scale base 10

    histogram_zorder: int = 1
    histogram_alpha: float = 0.5
    histogram_color: str = "DarkCyan"
    histogram_legend: str = "simulation"

    legend_labels: tuple[str] = (histogram_legend,)
    legend_handles: tuple[Patch] = (
        Patch(color=histogram_color, alpha=histogram_alpha),
    )

    def set_derived_attributes(self) -> None:
        self.compound: Compound = self.structure_simulation_arg("compound", Compound)

        self.legend: PartialWidthHistogramLegend = PartialWidthHistogramLegend(
            handles=self.legend_handles, labels=self.legend_labels
        )
        if self.legend.title is None:
            self.legend.title = self.compound.to_latex

        self.scale_limits_and_ticks(
            x=lambda value: 10**value,
            y=lambda value: 10**value,
        )

    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        self.create_figure()

        self.ax.set_xscale("log", base=10)
        self.ax.set_yscale("log", base=10)

        self.ax.xaxis.set_minor_formatter(NullFormatter())
        self.ax.yaxis.set_minor_formatter(NullFormatter())

        self.draw_histogram(
            color=self.histogram_color,
            alpha=self.histogram_alpha,
            zorder=self.histogram_zorder,
        )

        centers = rmtpy.density.compute_bin_centers(self.data.bins)

        self.ax.plot(
            centers,
            self.compound.ensemble.porter_thomas_distribution(1, centers),
            color="Black",
        )

        self.finish_plot(path=path)
