from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter

from ..._histogram import Histogram2D
from ..._plot import PlotAxes, PlotLegend, Plot
from ....compounds import Compound
from ....utils import rmtpy_converter


@dataclass(repr=False, eq=False, kw_only=True)
class UnfoldedComplexEnergyHistogramLegend(PlotLegend):
    loc: str = "upper right"
    bbox: tuple[float, float] = (0.94, 0.95)


@dataclass(repr=False, eq=False, kw_only=True)
class UnfoldedComplexEnergyHistogramAxes(PlotAxes):
    xticks: tuple[float, ...] = (-1.0, 0.0, 1.0)  # units of energy_0
    xticks_minor: tuple[float, ...] = (-0.5, 0.5)
    xlabel: str = r"$E / E_0$"
    xtick_labels: tuple[str, ...] = (
        r"$-1$",
        r"$0$",
        r"$+1$",
    )

    yticks: tuple[float, ...] = tuple(range(-4, 5, 2))  # log scale base 10
    yticks_minor: tuple[float, ...] = tuple()
    ylabel: str = r"$\gamma$"
    ytick_labels: tuple[str, ...] = (
        r"$10^{-4}$",
        r"$10^{-2}$",
        r"$10^{0}$",
        r"$10^{2}$",
        r"$10^{4}$",
    )


@dataclass(repr=False, eq=False, kw_only=True)
class UnfoldedComplexEnergyHistogramPlot(Plot):
    data: Histogram2D
    axes: UnfoldedComplexEnergyHistogramAxes = field(
        default_factory=UnfoldedComplexEnergyHistogramAxes
    )
    num_points: int = 1000

    xlim: tuple[float, float] = (-1.2, 1.2)  # units of energy_0
    ylim: tuple[float, float] = (-5.0, 5.0)  # log scale base 10

    histogram_zorder: int = 1
    histogram_alpha: float = 1.0
    histogram_color: str = "OrangeRed"
    histogram_legend: str = "simulation"

    legend_labels: tuple[str] = None
    legend_handles: tuple[Patch] = None

    def set_derived_attributes(self) -> None:
        try:
            compound_meta: dict = self.data.metadata["simulation"]["args"]["compound"]
        except KeyError:
            raise ValueError("Compound metadata not found.")
        except TypeError:
            raise ValueError("Metadata is not properly structured.")

        self.compound: Compound = rmtpy_converter.structure(compound_meta, Compound)

        self.legend: UnfoldedComplexEnergyHistogramLegend = (
            UnfoldedComplexEnergyHistogramLegend(
                handles=self.legend_handles, labels=self.legend_labels
            )
        )
        if self.legend.title is None:
            self.legend.title = self.compound.to_latex

        axes: UnfoldedComplexEnergyHistogramAxes = self.axes
        self.xlim = tuple(x for x in self.xlim)
        self.ylim = tuple(10**y for y in self.ylim)

        axes.xticks = tuple(xtick for xtick in axes.xticks)
        axes.yticks = tuple(10**ytick for ytick in axes.yticks)
        axes.xticks_minor = tuple(xtick for xtick in axes.xticks_minor)
        axes.yticks_minor = tuple(10**ytick for ytick in axes.yticks_minor)

    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        self.create_figure()

        self.ax.set_xscale("linear")
        self.ax.set_yscale("log", base=10)

        self.ax.yaxis.set_minor_formatter(NullFormatter())

        histogram = self.data.histogram.copy()
        histogram[histogram == 0] = np.nan

        self.ax.set_facecolor("Black")
        self.ax.tick_params(axis="both", which="both", color="White")

        x_mesh, y_mesh = np.meshgrid(self.data.x_bins, self.data.y_bins, indexing="ij")
        self.ax.pcolormesh(
            x_mesh,
            y_mesh,
            histogram,
            shading="flat",
            cmap="magma",
            norm=LogNorm(vmin=np.nanmin(histogram), vmax=np.nanmax(histogram)),
            alpha=self.histogram_alpha,
            zorder=self.histogram_zorder,
        )
        self.finish_plot(path=path)
