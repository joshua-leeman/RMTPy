from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter

from ....compounds import Compound
from ....conversion import rmtpy_converter
from ....density import compute_histogram_bin_centers
from ...histogram import Histogram
from ...plot import Plot, PlotAxes, PlotLegend

@dataclass(repr=False, eq=False, kw_only=True)
class TotalWidthHistogramLegend(PlotLegend):
    loc: str = "upper right"
    bbox: tuple[float, float] = (0.94, 0.95)


@dataclass(repr=False, eq=False, kw_only=True)
class TotalWidthHistogramAxes(PlotAxes):
    xticks: tuple[float, ...] = tuple(range(-1, 2))  # log scale base 10
    xticks_minor: tuple[float, ...] = tuple()
    xlabel: str = r"$Y = \Gamma / \ensavg{\Gamma}$"
    xtick_labels: tuple[str, ...] = (
        r"$10^{-1}$",
        r"$10^{0}$",
        r"$10^{1}$",
    )

    yticks: tuple[float, ...] = tuple(range(-3, 2))  # log scale base 10
    yticks_minor: tuple[float, ...] = tuple()
    ylabel: str = r"$P(Y)$"
    ytick_labels: tuple[str, ...] = (
        r"$10^{-3}$",
        r"$10^{-2}$",
        r"$10^{-1}$",
        r"$10^{0}$",
        r"$10^{1}$",
    )


@dataclass(repr=False, eq=False, kw_only=True)
class TotalWidthHistogramPlot(Plot):
    data: Histogram
    axes: TotalWidthHistogramAxes = field(default_factory=TotalWidthHistogramAxes)

    xlim: tuple[float, float] = (-1.2, 1.2)  # log scale base 10
    ylim: tuple[float, float] = (-3.2, 1.2)  # log scale base 10

    histogram_zorder: int = 1
    histogram_alpha: float = 0.5
    histogram_color: str = "BlueViolet"
    histogram_legend: str = "simulation"

    legend_labels: tuple[str] = (histogram_legend,)
    legend_handles: tuple[Patch] = (
        Patch(color=histogram_color, alpha=histogram_alpha),
    )

    def set_derived_attributes(self) -> None:
        try:
            compound_meta: dict = self.data.metadata["simulation"]["args"]["compound"]
        except KeyError:
            raise ValueError("Compound metadata not found.")
        except TypeError:
            raise ValueError("Metadata is not properly structured.")

        self.compound: Compound = rmtpy_converter.structure(compound_meta, Compound)

        self.legend: TotalWidthHistogramLegend = TotalWidthHistogramLegend(
            handles=self.legend_handles, labels=self.legend_labels
        )
        if self.legend.title is None:
            self.legend.title = self.compound.to_latex

        axes: TotalWidthHistogramAxes = self.axes
        self.xlim = tuple(10**x for x in self.xlim)
        self.ylim = tuple(10**y for y in self.ylim)

        axes.xticks = tuple(10**xtick for xtick in axes.xticks)
        axes.yticks = tuple(10**ytick for ytick in axes.yticks)
        axes.xticks_minor = tuple(10**xtick for xtick in axes.xticks_minor)
        axes.yticks_minor = tuple(10**ytick for ytick in axes.yticks_minor)

    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        self.create_figure()

        self.ax.set_xscale("log", base=10)
        self.ax.set_yscale("log", base=10)

        self.ax.xaxis.set_minor_formatter(NullFormatter())
        self.ax.yaxis.set_minor_formatter(NullFormatter())

        centers = compute_histogram_bin_centers(self.data.bins, bins_log_spaced=True)

        self.ax.hist(
            self.data.bins[:-1],
            bins=self.data.bins,
            weights=self.data.histogram,
            color=self.histogram_color,
            alpha=self.histogram_alpha,
            zorder=self.histogram_zorder,
        )

        ensemble = self.compound.ensemble
        num_channels = self.compound.num_channels

        self.ax.plot(
            centers,
            ensemble.porter_thomas_distribution(num_channels, centers),
            color="Black",
        )

        self.finish_plot(path=path)
