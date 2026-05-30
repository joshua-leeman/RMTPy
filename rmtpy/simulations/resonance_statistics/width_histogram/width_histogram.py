from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter

from ....compounds import Compound
from ....ensembles import ManyBodyEnsemble
from ...histogram import Histogram
from ...plot import Plot, PlotAxes, PlotLegend


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class WidthHistogramLegend(PlotLegend):
    loc: str = "upper right"
    bbox: tuple[float, float] = (0.94, 0.95)


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
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


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class WidthHistogramPlot(Plot):
    data: Histogram
    axes: WidthHistogramAxes = dataclasses.field(default_factory=WidthHistogramAxes)

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
        self.compound: Compound = self.structure_simulation_arg("compound", Compound)
        mean_coupling_squared: float = np.mean(
            self.compound.channel_coupling_strengths**2
        )
        ensemble: ManyBodyEnsemble = self.compound.ensemble
        energy_0: float = ensemble.spectral_radius

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

        self.finish_plot(path=path)


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class UnfoldedWidthHistogramLegend(PlotLegend):
    loc: str = "upper right"
    bbox: tuple[float, float] = (0.94, 0.95)


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class UnfoldedWidthHistogramAxes(PlotAxes):
    xticks: tuple[float, ...] = tuple(range(-4, 5, 2))  # log scale base 10
    xticks_minor: tuple[float, ...] = tuple(range(-3, 4, 2))
    xlabel: str = r"$\gamma$"
    xtick_labels: tuple[str, ...] = (
        r"$10^{-4}$",
        r"$10^{-2}$",
        r"$10^{0}$",
        r"$10^{2}$",
        r"$10^{4}$",
    )

    yticks: tuple[float, ...] = tuple(range(-4, 5, 2))  # log scale base 10
    yticks_minor: tuple[float, ...] = tuple(range(-3, 4, 2))
    ylabel: str = r"$\diff {P} / \diff \log \gamma$"
    ytick_labels: tuple[str, ...] = (
        r"$10^{-4}$",
        r"$10^{-2}$",
        r"$10^{0}$",
        r"$10^{2}$",
        r"$10^{4}$",
    )


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class UnfoldedWidthHistogramPlot(Plot):
    data: Histogram
    axes: UnfoldedWidthHistogramAxes = dataclasses.field(
        default_factory=UnfoldedWidthHistogramAxes
    )

    xlim: tuple[float, float] = (-3.5, 2.5)  # log scale base 10
    ylim: tuple[float, float] = (-4.5, 4.5)  # log scale base 10

    histogram_zorder: int = 1
    histogram_alpha: float = 0.5
    histogram_color: str = "SeaGreen"
    histogram_legend: str = "simulation"

    legend_labels: tuple[str] = (histogram_legend,)
    legend_handles: tuple[Patch] = (
        Patch(color=histogram_color, alpha=histogram_alpha),
    )

    def set_derived_attributes(self) -> None:
        self.compound: Compound = self.structure_simulation_arg("compound", Compound)
        mean_coupling_squared: float = np.mean(
            self.compound.channel_coupling_strengths**2
        )
        ensemble: ManyBodyEnsemble = self.compound.ensemble

        self.legend: UnfoldedWidthHistogramLegend = UnfoldedWidthHistogramLegend(
            handles=self.legend_handles, labels=self.legend_labels
        )
        ten_exponent: float = np.log10(mean_coupling_squared / ensemble.spectral_radius)
        ten_exponent = 0.00 if np.isclose(ten_exponent, 0) else ten_exponent
        if self.legend.title is None:
            self.legend.title = (
                self.compound.to_latex
                + "\n"
                + r"$\nu^2 = "
                + f"10^{{{ten_exponent:.2f}}}E_0$"
                + "\nunfolded"
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

        self.draw_histogram(
            color=self.histogram_color,
            alpha=self.histogram_alpha,
            zorder=self.histogram_zorder,
        )

        self.finish_plot(path=path)
