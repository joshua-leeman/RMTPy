import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import cast, override

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullLocator

from .... import universal
from ....compounds import CompoundEnsemble
from ...base_data import Data
from ...base_plot import (
    LogDimensionTimeAxes,
    LogDimensionUnfoldedTimeAxes,
    Plot,
    PlotAxes,
    PlotLegend,
)
from ...statistics import LOG_D_TIME_SUPPORT, LOG_D_UNFOLDED_TIME_SUPPORT
from .time_delay_histogram_data import TimeDelayHistogram


def format_energy_label(energy: float, energy_scale: float) -> str:
    scaled_energy = energy / energy_scale
    if np.isclose(scaled_energy, 0.0):
        return r"$E = 0$"

    return rf"$E = {scaled_energy:.2f}E_0$"


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class TimeDelayHistogramAxes(LogDimensionTimeAxes):
    ylabel: str = r"$P(u)$"


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class TimeDelayHistogramPlot(Plot):
    data: Data
    axes: PlotAxes = dataclasses.field(default_factory=TimeDelayHistogramAxes)
    num_points: int = 1000

    xlim: tuple[float, float] = LOG_D_TIME_SUPPORT

    histogram_zorder: int = 1
    histogram_alpha: float = 0.42
    histogram_color: str = "#7b2d26"

    pdf_zorder: int = 2
    pdf_width: float = 2.0
    pdf_alpha: float = 1.0
    pdf_color: str = "Black"
    pdf_legend: str = "BFB"

    def set_derived_attributes(self) -> None:
        self.compound: CompoundEnsemble = self.store_manifest_arg(
            "compound", CompoundEnsemble
        )
        mean_coupling_squared = cast(
            float, cast(object, np.mean(self.compound.couplings**2))
        )
        ensemble = self.compound.ensemble

        energy = cast(float, self.data.metadata["energy"])
        energy_label = format_energy_label(energy, ensemble.spectral_radius)
        self.legend_labels: tuple[str, str] = (energy_label, self.pdf_legend)
        self.legend_handles: tuple[Patch, Line2D] = (
            Patch(color=self.histogram_color, alpha=self.histogram_alpha),
            Line2D(
                [0],
                [0],
                color=self.pdf_color,
                alpha=self.pdf_alpha,
                linewidth=self.pdf_width,
            ),
        )

        self.legend: PlotLegend = PlotLegend(
            handles=self.legend_handles,
            labels=self.legend_labels,
            loc="upper right",
            bbox=(0.98, 0.95),
        )

        coupling_exponent = cast(
            float, np.log10(mean_coupling_squared / ensemble.spectral_radius)
        )
        coupling_exponent = 0.0 if abs(coupling_exponent) < 0.005 else coupling_exponent
        coupling_label = rf"$\alpha = {{{coupling_exponent:.1f}}}$"
        if not self.legend.title:
            self.legend.title = (
                ensemble.to_latex
                + "\n"
                + rf"$N_\text{{f}} = {{{self.compound.num_free_complex_fermions}}}$"
                + f", {{{coupling_label}}}"
            )

        scale = cast(float, self.data.metadata["scale"])
        self.scale_limits_and_ticks(
            x=lambda value: cast(float, ensemble.dimension**value * scale),
        )

    @override
    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        if not isinstance(self.data, TimeDelayHistogram):
            raise ValueError("Data must be a `TimeDelayHistogram` instance")

        self.build_figure()

        set_xscale = cast(Callable[..., object], self.ax.set_xscale)
        _ = set_xscale("log", base=self.compound.ensemble.dimension)

        set_major_locator = cast(Callable[..., object], self.ax.xaxis.set_major_locator)
        _ = set_major_locator(
            LogLocator(
                base=self.compound.ensemble.dimension,
                numticks=len(self.axes.xticks),
            )
        )
        set_minor_locator = cast(Callable[..., object], self.ax.xaxis.set_minor_locator)
        _ = set_minor_locator(NullLocator())

        self.draw_histogram(
            color=self.histogram_color,
            alpha=self.histogram_alpha,
            zorder=self.histogram_zorder,
        )

        times = np.geomspace(*self.xlim, self.num_points)
        time_delay_pdf = self.compound.time_delay_pdf(times)

        plot = cast(Callable[..., object], self.ax.plot)
        _ = plot(
            times,
            time_delay_pdf,
            color=self.pdf_color,
            alpha=self.pdf_alpha,
            linewidth=self.pdf_width,
            zorder=self.pdf_zorder,
        )

        self.finish_plot(path=path)


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class UnfoldedTimeDelayHistogramAxes(LogDimensionUnfoldedTimeAxes):
    ylabel: str = r"$P(\upsilon)$"


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class UnfoldedTimeDelayHistogramPlot(Plot):
    data: Data
    axes: PlotAxes = dataclasses.field(default_factory=UnfoldedTimeDelayHistogramAxes)
    num_points: int = 1000

    xlim: tuple[float, float] = LOG_D_UNFOLDED_TIME_SUPPORT

    histogram_zorder: int = 1
    histogram_alpha: float = 0.42
    histogram_color: str = "#7b2d26"

    pdf_zorder: int = 2
    pdf_width: float = 2.0
    pdf_alpha: float = 1.0
    pdf_color: str = "Black"
    pdf_legend: str = "BFB"

    def set_derived_attributes(self) -> None:
        self.compound: CompoundEnsemble = self.store_manifest_arg(
            "compound", CompoundEnsemble
        )
        mean_coupling_squared = float(np.mean(self.compound.couplings**2))
        ensemble = self.compound.ensemble

        energy = cast(float, self.data.metadata["energy"])
        energy_label = format_energy_label(energy, ensemble.spectral_radius)
        self.legend_labels: tuple[str, str] = (energy_label, self.pdf_legend)
        self.legend_handles: tuple[Patch, Line2D] = (
            Patch(color=self.histogram_color, alpha=self.histogram_alpha),
            Line2D(
                [0],
                [0],
                color=self.pdf_color,
                alpha=self.pdf_alpha,
                linewidth=self.pdf_width,
            ),
        )

        self.legend: PlotLegend = PlotLegend(
            handles=self.legend_handles,
            labels=self.legend_labels,
            loc="upper right",
            bbox=(0.98, 0.95),
        )

        coupling_exponent = cast(
            float, np.log10(mean_coupling_squared / ensemble.spectral_radius)
        )
        coupling_exponent = 0.0 if abs(coupling_exponent) < 0.005 else coupling_exponent
        coupling_label = rf"$\alpha = {{{coupling_exponent:.1f}}}$"
        if not self.legend.title:
            unfolding_type = self.data.metadata["unfolding"]
            if unfolding_type != "weight":
                polynomial_degree = self.data.metadata["polynomial_degree"]
                self.legend.title = (
                    ensemble.to_latex
                    + f"\n{unfolding_type} unfolded, degree {polynomial_degree}"
                    + "\n"
                    + rf"$N_\text{{f}} = {{{self.compound.num_free_complex_fermions}}}$"
                    + f", {{{coupling_label}}}"
                )
            else:
                self.legend.title = (
                    ensemble.to_latex
                    + "\nweight unfolded"
                    + "\n"
                    + rf"$N_\text{{f}} = {{{self.compound.num_free_complex_fermions}}}$"
                    + f", {{{coupling_label}}}"
                )

        scale = cast(float, self.data.metadata["scale"])
        self.scale_limits_and_ticks(
            x=lambda value: cast(float, ensemble.dimension**value * scale),
        )

    @override
    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        if not isinstance(self.data, TimeDelayHistogram):
            raise ValueError("Data must be a `TimeDelayHistogram` instance")

        self.build_figure()

        set_xscale = cast(Callable[..., object], self.ax.set_xscale)
        _ = set_xscale("log", base=self.compound.ensemble.dimension)

        set_major_locator = cast(Callable[..., object], self.ax.xaxis.set_major_locator)
        _ = set_major_locator(
            LogLocator(
                base=self.compound.ensemble.dimension,
                numticks=len(self.axes.xticks),
            )
        )
        set_minor_locator = cast(Callable[..., object], self.ax.xaxis.set_minor_locator)
        _ = set_minor_locator(NullLocator())

        self.draw_histogram(
            color=self.histogram_color,
            alpha=self.histogram_alpha,
            zorder=self.histogram_zorder,
        )

        times = np.geomspace(*self.xlim, self.num_points)
        time_delay_pdf = universal.time_delay_pdf(
            times,
            num_channels=self.compound.num_channels,
            heisenberg_time=2 * np.pi,
        )

        plot = cast(Callable[..., object], self.ax.plot)
        _ = plot(
            times,
            time_delay_pdf,
            color=self.pdf_color,
            alpha=self.pdf_alpha,
            linewidth=self.pdf_width,
            zorder=self.pdf_zorder,
        )

        self.finish_plot(path=path)
