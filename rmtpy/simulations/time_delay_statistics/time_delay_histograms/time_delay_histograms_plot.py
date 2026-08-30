from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullLocator

import rmtpy.universal
from rmtpy.compounds import Compound

from ...histogram import Histogram
from ...plot import (
    DIMENSION_TIME_LOG_SUPPORT,
    UNFOLDED_DIMENSION_TIME_LOG_SUPPORT,
    DimensionTimeAxes,
    Plot,
    PlotLegend,
    UnfoldedDimensionTimeAxes,
)


def format_energy_label(energy: float, energy_0: float) -> str:
    scaled_energy = energy / energy_0
    if np.isclose(scaled_energy, 0.0):
        return r"$E = 0$"

    return rf"$E = {scaled_energy:.2f}E_0$"


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class TimeDelayHistogramAxes(DimensionTimeAxes):
    ylabel: str = r"$P(u)$"


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class TimeDelayHistogramPlot(Plot):
    data: Histogram
    axes: TimeDelayHistogramAxes = dataclasses.field(
        default_factory=TimeDelayHistogramAxes
    )
    num_points: int = 1000

    xlim: tuple[float, float] = DIMENSION_TIME_LOG_SUPPORT

    histogram_zorder: int = 1
    histogram_alpha: float = 0.42
    histogram_color: str = "#7b2d26"

    pdf_zorder: int = 2
    pdf_width: float = 2.0
    pdf_alpha: float = 1.0
    pdf_color: str = "Black"
    pdf_legend: str = "BFB"

    def set_derived_attributes(self) -> None:
        self.compound: Compound = self.structure_simulation_arg("compound", Compound)
        mean_coupling_squared = np.mean(self.compound.coupling_strengths**2)
        ensemble = self.compound.ensemble

        energy_0 = self.compound.ensemble.spectral_radius
        dimension = self.compound.ensemble.dimension
        energy = self.data.metadata["energy"]

        energy_label = format_energy_label(energy, energy_0)
        self.legend_labels: tuple[str, str] = (energy_label, self.pdf_legend)
        self.legend_handles: tuple[Patch, Line2D] = (
            Patch(color=self.histogram_color, alpha=self.histogram_alpha),
            Line2D([0], [0], color=self.pdf_color, linewidth=self.pdf_width),
        )

        self.legend = PlotLegend(
            handles=self.legend_handles,
            labels=self.legend_labels,
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

        self.scale_limits_and_ticks(
            x=lambda value: dimension**value * self.data.metadata["scale"],
        )

    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        self.create_figure()

        self.ax.set_xscale("log", base=self.compound.ensemble.dimension)
        self.ax.xaxis.set_major_locator(
            LogLocator(
                base=self.compound.ensemble.dimension,
                numticks=len(self.axes.xticks),
            )
        )
        self.ax.xaxis.set_minor_locator(NullLocator())

        self.draw_histogram(
            color=self.histogram_color,
            alpha=self.histogram_alpha,
            zorder=self.histogram_zorder,
        )

        times = np.geomspace(*self.xlim, self.num_points)
        time_delay_pdf = self.compound.time_delay_pdf(times=times)

        self.ax.plot(
            times,
            time_delay_pdf,
            color=self.pdf_color,
            alpha=self.pdf_alpha,
            linewidth=self.pdf_width,
            zorder=self.pdf_zorder,
        )

        self.finish_plot(path=path)


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class UnfoldedTimeDelayHistogramAxes(UnfoldedDimensionTimeAxes):
    ylabel: str = r"$P(\upsilon)$"


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class UnfoldedTimeDelayHistogramPlot(TimeDelayHistogramPlot):
    axes: UnfoldedTimeDelayHistogramAxes = dataclasses.field(
        default_factory=UnfoldedTimeDelayHistogramAxes
    )

    xlim: tuple[float, float] = UNFOLDED_DIMENSION_TIME_LOG_SUPPORT

    def set_derived_attributes(self) -> None:
        self.compound: Compound = self.structure_simulation_arg("compound", Compound)
        mean_coupling_squared = np.mean(self.compound.coupling_strengths**2)
        ensemble = self.compound.ensemble

        energy_0 = self.compound.ensemble.spectral_radius
        dimension = self.compound.ensemble.dimension
        energy = self.data.metadata["energy"]

        energy_label = format_energy_label(energy, energy_0)
        self.legend_labels: tuple[str, str] = (energy_label, self.pdf_legend)
        self.legend_handles: tuple[Patch, Line2D] = (
            Patch(color=self.histogram_color, alpha=self.histogram_alpha),
            Line2D([0], [0], color=self.pdf_color, linewidth=self.pdf_width),
        )

        self.legend = PlotLegend(
            handles=self.legend_handles,
            labels=self.legend_labels,
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

        self.scale_limits_and_ticks(
            x=lambda value: dimension**value * self.data.metadata["scale"],
        )

    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        self.create_figure()

        self.ax.set_xscale("log", base=self.compound.ensemble.dimension)
        self.ax.xaxis.set_major_locator(
            LogLocator(
                base=self.compound.ensemble.dimension,
                numticks=len(self.axes.xticks),
            )
        )
        self.ax.xaxis.set_minor_locator(NullLocator())

        self.draw_histogram(
            color=self.histogram_color,
            alpha=self.histogram_alpha,
            zorder=self.histogram_zorder,
        )

        times = np.geomspace(*self.xlim, self.num_points)
        time_delay_pdf = rmtpy.universal.time_delay_pdf(
            times,
            num_channels=self.compound.num_channels,
            heisenberg_time=2 * np.pi,
        )

        self.ax.plot(
            times,
            time_delay_pdf,
            color=self.pdf_color,
            alpha=self.pdf_alpha,
            linewidth=self.pdf_width,
            zorder=self.pdf_zorder,
        )

        self.finish_plot(path=path)
