from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter

from ....compounds import Compound
from ...histogram2D import Histogram2D
from ...plot import Plot, PlotAxes, PlotLegend


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class ComplexEnergyHistogramAxes(PlotAxes):
    xticks: tuple[float, ...] = (-1.0, 0.0, 1.0)  # units of energy_0
    xticks_minor: tuple[float, ...] = (-0.5, 0.5)
    xlabel: str = r"$E / E_0$"
    xtick_labels: tuple[str, ...] = (
        r"$-1.0$",
        r"$0.0$",
        r"$+1.0$",
    )

    yticks: tuple[float, ...] = tuple(range(-4, 5, 2))  # log scale base 10
    yticks_minor: tuple[float, ...] = tuple()
    ylabel: str = r"$\log_{10}(\Gamma / E_0)$"
    ytick_labels: tuple[str, ...] = (
        r"$-4.0$",
        r"$-2.0$",
        r"$0.0$",
        r"$+2.0$",
        r"$+4.0$",
    )


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class ComplexEnergyHistogramPlot(Plot):
    data: Histogram2D
    axes: ComplexEnergyHistogramAxes = dataclasses.field(
        default_factory=ComplexEnergyHistogramAxes
    )
    num_points: int = 1000

    xlim: tuple[float, float] = (-1.2, 1.2)  # units of energy_0
    ylim: tuple[float, float] = (-5.0, 5.0)  # log scale base 10

    histogram_zorder: int = 1
    histogram_alpha: float = 1.0
    histogram_color: str = "OrangeRed"
    histogram_legend: str = "simulation"

    width_curve_zorder: int = 2
    width_curve_alpha: float = 1.0
    width_curve_width: float = 1.0
    width_curve_color: str = "Cyan"
    width_curve_legend: str = "average width"

    legend_labels: tuple[str] = (histogram_legend, width_curve_legend)
    legend_handles: tuple[Patch] = (
        Patch(color=histogram_color, alpha=histogram_alpha),
        Line2D([0], [0], color=width_curve_color, linewidth=width_curve_width),
    )

    def set_derived_attributes(self) -> None:
        self.compound: Compound = self.structure_simulation_arg("compound", Compound)
        mean_coupling_squared = np.mean(self.compound.coupling_strengths**2)
        ensemble = self.compound.ensemble

        self.legend = PlotLegend(
            handles=self.legend_handles,
            labels=self.legend_labels,
            on_black_background=True,
            loc="upper right",
            bbox=(0.98, 0.95),
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

        self.scale_limits_and_ticks(y=lambda value: 10**value)

    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        self.create_figure()

        self.ax.set_xscale("linear")
        self.ax.set_yscale("log", base=10)

        self.ax.yaxis.set_minor_formatter(NullFormatter())

        self.ax.set_facecolor("Black")
        self.ax.tick_params(axis="both", which="both", color="White")

        histogram = self.data.histogram.copy()
        positive_values = histogram[histogram > 0.0]
        if positive_values.size:
            histogram[histogram == 0.0] = np.nan
            color_min = np.min(positive_values)
            color_max = np.max(positive_values)
            if color_min == color_max:
                color_max = np.nextafter(color_max, np.inf)

            x_mesh, y_mesh = np.meshgrid(
                self.data.x_bins,
                self.data.y_bins,
                indexing="ij",
            )
            self.ax.pcolormesh(
                x_mesh,
                y_mesh,
                histogram,
                shading="flat",
                cmap="magma",
                norm=LogNorm(vmin=color_min, vmax=color_max),
                alpha=self.histogram_alpha,
                zorder=self.histogram_zorder,
            )

            x_values, average_y_given_x = self.data.compute_average_x_curve()
            self.ax.plot(
                x_values,
                average_y_given_x,
                color=self.width_curve_color,
                alpha=self.width_curve_alpha,
                linewidth=self.width_curve_width,
                zorder=self.width_curve_zorder,
            )

        self.finish_plot(path=path)


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class UnfoldedComplexEnergyHistogramAxes(ComplexEnergyHistogramAxes):
    ylabel: str = r"$\log_{10}\gamma$"


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class UnfoldedComplexEnergyHistogramPlot(ComplexEnergyHistogramPlot):
    axes: UnfoldedComplexEnergyHistogramAxes = dataclasses.field(
        default_factory=UnfoldedComplexEnergyHistogramAxes
    )

    def set_derived_attributes(self) -> None:
        self.compound: Compound = self.structure_simulation_arg("compound", Compound)
        mean_coupling_squared = np.mean(self.compound.coupling_strengths**2)
        ensemble = self.compound.ensemble

        self.legend = PlotLegend(
            handles=self.legend_handles,
            labels=self.legend_labels,
            on_black_background=True,
            loc="upper right",
            bbox=(0.98, 0.95),
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

        self.scale_limits_and_ticks(y=lambda value: 10**value)
