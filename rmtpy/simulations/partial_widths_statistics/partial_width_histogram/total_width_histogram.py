from __future__ import annotations

import dataclasses
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter

import rmtpy.density
from rmtpy.compounds import Compound

from ...histogram import Histogram
from ...plot import Plot, PlotAxes, PlotLegend


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class TotalWidthHistogramAxes(PlotAxes):
    xticks: tuple[float, ...] = tuple(range(-1, 2))  # log scale base 10
    xticks_minor: tuple[float, ...] = tuple()
    xlabel: str = r"$\log_{10} Y$"
    xtick_labels: tuple[str, ...] = (
        r"$-1.0$",
        r"$0.0$",
        r"$+1.0$",
    )

    yticks: tuple[float, ...] = tuple(range(-3, 2))  # log scale base 10
    yticks_minor: tuple[float, ...] = tuple()
    ylabel: str = r"$\log_{10} P(Y)$"
    ytick_labels: tuple[str, ...] = (
        r"$-3.0$",
        r"$-2.0$",
        r"$-1.0$",
        r"$0.0$",
        r"$+1.0$",
    )


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class TotalWidthHistogramPlot(Plot):
    data: Histogram
    axes: TotalWidthHistogramAxes = dataclasses.field(
        default_factory=TotalWidthHistogramAxes
    )

    xlim: tuple[float, float] = (-1.2, 1.2)  # log scale base 10
    ylim: tuple[float, float] = (-3.2, 1.2)  # log scale base 10

    histogram_zorder: int = 1
    histogram_alpha: float = 0.5
    histogram_color: str = "BlueViolet"
    histogram_legend: str = "simulation"

    surmise_zorder: int = 2
    surmise_width: float = 2.0
    surmise_alpha: float = 1.0
    surmise_color: str = "Black"
    surmise_legend: str = "Porter-Thomas"

    legend_labels: tuple[str] = (histogram_legend, surmise_legend)
    legend_handles: tuple[Patch] = (
        Patch(color=histogram_color, alpha=histogram_alpha),
        Line2D([0], [0], color=surmise_color, linewidth=surmise_width),
    )

    def set_derived_attributes(self) -> None:
        width_index: tuple[int, int] = self.data.metadata["index"]
        state_label = width_index[0]

        self.axes.xlabel = rf"$\log_{{10}} Y_{{{state_label}}}$"
        self.axes.ylabel = rf"$\log_{{10}} P(Y_{{{state_label}}})$"

        self.compound: Compound = self.structure_simulation_arg("compound", Compound)

        self.legend = PlotLegend(
            handles=self.legend_handles,
            labels=self.legend_labels,
            loc="upper right",
            bbox=(0.97, 0.95),
        )

        if self.legend.title is None:
            self.legend.title = (
                self.compound.ensemble.to_latex
                + "\n"
                + rf"$N_\text{{f}} = {{{self.compound.num_free_complex_fermions}}}$"
                + rf", $\mu = {{{state_label}}}$"
            )

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

        centers = rmtpy.density.compute_bin_centers(self.data.bins)

        self.draw_histogram(
            color=self.histogram_color,
            alpha=self.histogram_alpha,
            zorder=self.histogram_zorder,
        )

        ensemble = self.compound.ensemble
        num_channels = self.compound.num_channels

        self.ax.plot(
            centers,
            ensemble.porter_thomas_distribution(
                centers,
                num_channels=num_channels,
            ),
            color="Black",
        )

        self.finish_plot(path=path)
