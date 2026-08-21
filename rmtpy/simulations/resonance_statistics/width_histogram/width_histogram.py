from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter

from ....compounds import Compound
from ...histogram import Histogram
from ...plot import Plot, PlotAxes, PlotLegend


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class WidthHistogramAxes(PlotAxes):
    xticks: tuple[float, ...] = tuple(range(-3, 4))  # log scale base 10
    xticks_minor: tuple[float, ...] = tuple()
    xlabel: str = r"$x = \log_{10} (\Gamma / E_0)$"
    xtick_labels: tuple[str, ...] = (
        r"$-3.0$",
        r"$-2.0$",
        r"$-1.0$",
        r"$0.0$",
        r"$+1.0$",
        r"$+2.0$",
        r"$+3.0$",
    )

    yticks: tuple[float, ...] = tuple(range(-3, 4))  # log scale base 10
    yticks_minor: tuple[float, ...] = tuple()
    ylabel: str = r"$P(x)$"
    ytick_labels: tuple[str, ...] = (
        r"$-3.0$",
        r"$-2.0$",
        r"$-1.0$",
        r"$0.0$",
        r"$+1.0$",
        r"$+2.0$",
        r"$+3.0$",
    )


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class WidthHistogramPlot(Plot):
    data: Histogram
    axes: WidthHistogramAxes = dataclasses.field(default_factory=WidthHistogramAxes)

    xlim: tuple[float, float] = (-3.5, 3.5)  # log scale base 10
    ylim: tuple[float, float] = (-3.5, 3.5)  # log scale base 10

    histogram_zorder: int = 1
    histogram_alpha: float = 0.5
    histogram_color: str = "SeaGreen"
    histogram_legend: str = "simulation"

    legend_labels: tuple[str] = (histogram_legend,)
    legend_handles: tuple[Patch] = (Patch(color=histogram_color, alpha=histogram_alpha),)

    def set_derived_attributes(self) -> None:
        self.compound: Compound = self.structure_simulation_arg("compound", Compound)
        mean_coupling_squared = np.mean(self.compound.coupling_strengths**2)
        ensemble = self.compound.ensemble

        self.legend = PlotLegend(
            handles=self.legend_handles,
            labels=self.legend_labels,
            loc="upper right",
            bbox=(0.94, 0.95),
        )

        coupling_exponent = np.log10(mean_coupling_squared / ensemble.spectral_radius)
        coupling_exponent = 0.0 if abs(coupling_exponent) < 0.005 else coupling_exponent
        coupling_label = rf"$\alpha = {{{coupling_exponent:.1f}}}$"
        if self.legend.title is None:
            self.legend.title = (
                self.compound.ensemble.to_latex
                + "\n"
                + rf"$N_\text{{f}} = {{{self.compound.num_free_complex_fermions}}}$"
                + f", {{{coupling_label}}}"
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
class UnfoldedWidthHistogramAxes(PlotAxes):
    xticks: tuple[float, ...] = tuple(range(-3, 4))  # log scale base 10
    xticks_minor: tuple[float, ...] = tuple(range(-2, 3))
    xlabel: str = r"$\zeta = \log_{10} \gamma$"
    xtick_labels: tuple[str, ...] = (
        r"$-3.0$",
        r"$-2.0$",
        r"$-1.0$",
        r"$0.0$",
        r"$+1.0$",
        r"$+2.0$",
        r"$+3.0$",
    )

    yticks: tuple[float, ...] = tuple(range(-3, 4))  # log scale base 10
    yticks_minor: tuple[float, ...] = tuple(range(-2, 3))
    ylabel: str = r"$P(\zeta)$"
    ytick_labels: tuple[str, ...] = (
        r"$-3.0$",
        r"$-2.0$",
        r"$-1.0$",
        r"$0.0$",
        r"$+1.0$",
        r"$+2.0$",
        r"$+3.0$",
    )


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class UnfoldedWidthHistogramPlot(Plot):
    data: Histogram
    axes: UnfoldedWidthHistogramAxes = dataclasses.field(
        default_factory=UnfoldedWidthHistogramAxes
    )

    xlim: tuple[float, float] = (-3.5, 3.5)  # log scale base 10
    ylim: tuple[float, float] = (-3.5, 3.5)  # log scale base 10

    histogram_zorder: int = 1
    histogram_alpha: float = 0.5
    histogram_color: str = "SeaGreen"
    histogram_legend: str = "simulation"

    legend_labels: tuple[str] = (histogram_legend,)
    legend_handles: tuple[Patch] = (Patch(color=histogram_color, alpha=histogram_alpha),)

    def set_derived_attributes(self) -> None:
        self.compound: Compound = self.structure_simulation_arg("compound", Compound)
        mean_coupling_squared = np.mean(self.compound.coupling_strengths**2)
        ensemble = self.compound.ensemble

        self.legend = PlotLegend(
            handles=self.legend_handles,
            labels=self.legend_labels,
            loc="upper right",
            bbox=(0.94, 0.95),
        )

        coupling_exponent = np.log10(mean_coupling_squared / ensemble.spectral_radius)
        coupling_exponent = 0 if abs(coupling_exponent) < 0.005 else coupling_exponent
        coupling_label = rf"$\alpha = {{{coupling_exponent:.1f}}}$"
        if self.legend.title is None:
            unfolding_type = self.data.metadata["unfolding"]
            if unfolding_type != "wgt":
                unfolding_degree = self.data.metadata["degree"]
                self.legend.title = (
                    self.compound.ensemble.to_latex
                    + f"\n{unfolding_type}.\ unfolded, degree {unfolding_degree}"
                    + "\n"
                    + rf"$N_\text{{f}} = {{{self.compound.num_free_complex_fermions}}}$"
                    + f", {{{coupling_label}}}"
                )
            else:
                self.legend.title = (
                    self.compound.ensemble.to_latex
                    + "\nwgt.\ unfolded"
                    + "\n"
                    + rf"$N_\text{{f}} = {{{self.compound.num_free_complex_fermions}}}$"
                    + f", {{{coupling_label}}}"
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
