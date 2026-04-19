from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter

from ..._histogram import Histogram
from ..._plot import PlotAxes, PlotLegend, Plot
from ....compounds import Compound
from ....ensembles import ManyBodyEnsemble
from ....utils import rmtpy_converter


@dataclass(repr=False, eq=False, kw_only=True)
class WidthHistogramLegend(PlotLegend):
    loc: str = "upper right"
    bbox: tuple[float, float] = (0.94, 0.95)


@dataclass(repr=False, eq=False, kw_only=True)
class WidthHistogramAxes(PlotAxes):
    xticks: tuple[float, ...] = tuple(range(-3, 4))  # log scale base 10
    xticks_minor: tuple[float, ...] = tuple()
    xlabel: str = r"$\Gamma / E_0$"
    xtick_labels: tuple[str, ...] = (
        r"$10^{-3}$",
        r"$10^{-2}$",
        r"$10^{-1}$",
        r"$10^{0}$",
        r"$10^{1}$",
        r"$10^{2}$",
        r"$10^{3}$",
    )

    yticks: tuple[float, ...] = tuple(range(-4, 4))  # log scale base 10
    yticks_minor: tuple[float, ...] = tuple()
    ylabel: str = r"$\diff {P} / \diff \log (\Gamma / E_0)$"
    ytick_labels: tuple[str, ...] = (
        r"$10^{-4}$",
        r"$10^{-3}$",
        r"$10^{-2}$",
        r"$10^{-1}$",
        r"$10^{0}$",
        r"$10^{1}$",
        r"$10^{2}$",
        r"$10^{3}$",
    )


@dataclass(repr=False, eq=False, kw_only=True)
class WidthHistogramPlot(Plot):
    data: Histogram
    axes: WidthHistogramAxes = field(default_factory=WidthHistogramAxes)

    xlim: tuple[float, float] = (-3.2, 3.2)  # log scale base 10
    ylim: tuple[float, float] = (-4.2, 3.2)  # log scale base 10

    histogram_zorder: int = 1
    histogram_alpha: float = 0.5
    histogram_color: str = "SeaGreen"
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
        mean_coupling_squared: float = np.mean(
            self.compound.channel_coupling_strengths**2
        )
        ensemble: ManyBodyEnsemble = self.compound.ensemble
        energy_0: float = ensemble.ground_state_energy

        self.legend: WidthHistogramLegend = WidthHistogramLegend(
            handles=self.legend_handles, labels=self.legend_labels
        )
        if self.legend.title is None:
            self.legend.title = (
                self.compound.to_latex
                + "\n"
                + r"$\nu^2 = "
                + f"10^{{{np.log10(mean_coupling_squared / energy_0):.2f}}}E_0$"
            )

        axes: WidthHistogramAxes = self.axes
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

        centers = np.sqrt(self.data.bins[:-1] * self.data.bins[1:])
        log_weights = self.data.histogram * centers

        self.ax.hist(
            self.data.bins[:-1],
            bins=self.data.bins,
            weights=log_weights,
            color=self.histogram_color,
            alpha=self.histogram_alpha,
            zorder=self.histogram_zorder,
        )

        self.finish_plot(path=path)
