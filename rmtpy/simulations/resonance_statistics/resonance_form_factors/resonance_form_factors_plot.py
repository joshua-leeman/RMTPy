import dataclasses
import math
from collections.abc import Callable
from pathlib import Path
from typing import cast, override

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator
from scipy.special import jn_zeros

from ....compounds import CompoundEnsemble
from ...base_data import Data
from ...base_plot import (
    LogDimensionTimeAxes,
    LogDimensionUnfoldedTimeAxes,
    Plot,
    PlotAxes,
    PlotLegend,
)
from ...spectral_statistics.spectral_form_factors import FormFactorsData
from ...statistics import LOG_D_TIME_SUPPORT, LOG_D_UNFOLDED_TIME_SUPPORT


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class ResonanceFormFactorsAxes(LogDimensionTimeAxes):
    yticks: tuple[float, ...] = (-2, -1, 0)  # log scale base dimension
    ytick_labels: tuple[str, ...] = (
        r"$D^{-2}$",
        r"$D^{-1}$",
        r"$1$",
    )


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class ResonanceFormFactorsPlot(Plot):
    data: Data
    axes: PlotAxes = dataclasses.field(default_factory=ResonanceFormFactorsAxes)
    num_points: int = 1000

    xlim: tuple[float, float] = LOG_D_TIME_SUPPORT
    ylim: tuple[float, float] = (-2.2, 0.2)

    sff_zorder: int = 2
    sff_width: float = 0.5
    sff_alpha: float = 1.0
    sff_color: str = "Blue"
    sff_legend: str = r"$K(u)$"

    csff_zorder: int = 2
    csff_width: float = 0.5
    csff_alpha: float = 1.0
    csff_color: str = "Red"
    csff_legend: str = r"$K_{\text{\tiny conn}}(u)$"

    legend_labels: tuple[str, str] = (sff_legend, csff_legend)
    legend_handles: tuple[Line2D, Line2D] = (
        Line2D([0], [0], color=sff_color, alpha=sff_alpha, linewidth=sff_width),
        Line2D([0], [0], color=csff_color, alpha=csff_alpha, linewidth=csff_width),
    )

    def set_derived_attributes(self) -> None:
        self.compound: CompoundEnsemble = self.store_manifest_arg(
            "compound", CompoundEnsemble
        )
        mean_coupling_squared = float(np.mean(self.compound.couplings**2))
        ensemble = self.compound.ensemble

        self.legend: PlotLegend = PlotLegend(
            handles=self.legend_handles,
            labels=self.legend_labels,
            loc="upper right",
            bbox=(0.925, 0.95),
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

        j_1_1 = cast(float, jn_zeros(1, 1)[0])
        self.scale_limits_and_ticks(
            x=lambda value: (
                math.pow(ensemble.dimension, value) * j_1_1 / ensemble.spectral_radius
            ),
            y=lambda value: math.pow(ensemble.dimension, value),
        )

    @override
    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        if not isinstance(self.data, FormFactorsData):
            raise ValueError("Data must be a `FormFactorsData` instance")

        self.build_figure()

        dimension = self.compound.ensemble.dimension

        set_xscale = cast(Callable[..., object], self.ax.set_xscale)
        _ = set_xscale("log", base=dimension)

        set_yscale = cast(Callable[..., object], self.ax.set_yscale)
        _ = set_yscale("log", base=dimension)

        set_x_major_locator = cast(Callable[..., object], self.ax.xaxis.set_major_locator)
        _ = set_x_major_locator(
            LogLocator(base=dimension, numticks=len(self.axes.xticks))
        )

        set_y_major_locator = cast(Callable[..., object], self.ax.yaxis.set_major_locator)
        _ = set_y_major_locator(
            LogLocator(base=dimension, numticks=len(self.axes.yticks))
        )

        plot = cast(Callable[..., object], self.ax.plot)
        _ = plot(
            self.data.times,
            self.data.form_factor,
            color=self.sff_color,
            alpha=self.sff_alpha,
            linewidth=self.sff_width,
            zorder=self.sff_zorder,
            label=self.sff_legend,
        )
        _ = plot(
            self.data.times,
            self.data.connected_form_factor,
            color=self.csff_color,
            alpha=self.csff_alpha,
            linewidth=self.csff_width,
            zorder=self.csff_zorder,
            label=self.csff_legend,
        )

        self.finish_plot(path=path)


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class UnfoldedResonanceFormFactorsAxes(LogDimensionUnfoldedTimeAxes):
    yticks: tuple[float, ...] = (-2, -1, 0)  # log scale base dimension
    ytick_labels: tuple[str, ...] = (
        r"$D^{-2}$",
        r"$D^{-1}$",
        r"$1$",
    )


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class UnfoldedResonanceFormFactorsPlot(Plot):
    data: Data
    axes: PlotAxes = dataclasses.field(default_factory=UnfoldedResonanceFormFactorsAxes)
    num_points: int = 1000

    xlim: tuple[float, float] = LOG_D_UNFOLDED_TIME_SUPPORT
    ylim: tuple[float, float] = (-2.2, 0.2)

    sff_zorder: int = 2
    sff_width: float = 0.5
    sff_alpha: float = 1.0
    sff_color: str = "Blue"
    sff_legend: str = r"$K(\upsilon)$"

    csff_zorder: int = 2
    csff_width: float = 0.5
    csff_alpha: float = 1.0
    csff_color: str = "Red"
    csff_legend: str = r"$K_{\text{\tiny conn}}(\upsilon)$"

    universal_sff_zorder: int = 2
    universal_sff_width: float = 0.5
    universal_sff_alpha: float = 1.0
    universal_sff_color: str = "Black"
    universal_sff_legend: str = "universal"

    legend_labels: tuple[str, str, str] = (
        sff_legend,
        csff_legend,
        universal_sff_legend,
    )
    legend_handles: tuple[Line2D, Line2D, Line2D] = (
        Line2D([0], [0], color=sff_color, alpha=sff_alpha, linewidth=sff_width),
        Line2D([0], [0], color=csff_color, alpha=csff_alpha, linewidth=csff_width),
        Line2D(
            [0],
            [0],
            color=universal_sff_color,
            alpha=universal_sff_alpha,
            linewidth=universal_sff_width,
        ),
    )

    def set_derived_attributes(self) -> None:
        self.compound: CompoundEnsemble = self.store_manifest_arg(
            "compound", CompoundEnsemble
        )
        mean_coupling_squared = float(np.mean(self.compound.couplings**2))
        ensemble = self.compound.ensemble

        if ensemble.universality_class is not None:
            self.universal_sff_legend = (
                rf"$K^{{\text{{\tiny {ensemble.universality_class}}}}}"
                + r"_{\text{\tiny conn}}(\upsilon)$"
            )
            self.legend_labels = (
                self.sff_legend,
                self.csff_legend,
                self.universal_sff_legend,
            )

        self.legend: PlotLegend = PlotLegend(
            handles=self.legend_handles,
            labels=self.legend_labels,
            loc="upper right",
            bbox=(0.925, 0.95),
        )

        coupling_exponent = cast(
            float, np.log10(mean_coupling_squared / ensemble.spectral_radius)
        )
        coupling_exponent = 0.0 if abs(coupling_exponent) < 0.005 else coupling_exponent
        coupling_label = rf"$\alpha = {{{coupling_exponent:.1f}}}$"
        if not self.legend.title:
            unfolding_type = self.data.metadata["unfolding"]
            if unfolding_type != "weight":
                unfolding_degree = self.data.metadata["polynomial_degree"]
                self.legend.title = (
                    ensemble.to_latex
                    + f"\n{unfolding_type} unfolded, degree {unfolding_degree}"
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

        self.scale_limits_and_ticks(
            x=lambda value: math.pow(ensemble.dimension, value) * 2 * np.pi,
            y=lambda value: math.pow(ensemble.dimension, value),
        )

    @override
    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        if not isinstance(self.data, FormFactorsData):
            raise ValueError("Data must be a `FormFactorsData` instance")

        self.build_figure()

        dimension = self.compound.ensemble.dimension

        set_xscale = cast(Callable[..., object], self.ax.set_xscale)
        _ = set_xscale("log", base=dimension)

        set_yscale = cast(Callable[..., object], self.ax.set_yscale)
        _ = set_yscale("log", base=dimension)

        set_x_major_locator = cast(Callable[..., object], self.ax.xaxis.set_major_locator)
        _ = set_x_major_locator(
            LogLocator(base=dimension, numticks=len(self.axes.xticks))
        )

        set_y_major_locator = cast(Callable[..., object], self.ax.yaxis.set_major_locator)
        _ = set_y_major_locator(
            LogLocator(base=dimension, numticks=len(self.axes.yticks))
        )

        plot = cast(Callable[..., object], self.ax.plot)
        _ = plot(
            self.data.times,
            self.data.form_factor,
            color=self.sff_color,
            alpha=self.sff_alpha,
            linewidth=self.sff_width,
            zorder=self.sff_zorder,
            label=self.sff_legend,
        )
        _ = plot(
            self.data.times,
            self.data.connected_form_factor,
            color=self.csff_color,
            alpha=self.csff_alpha,
            linewidth=self.csff_width,
            zorder=self.csff_zorder,
            label=self.csff_legend,
        )

        universal_sff = self.compound.ensemble.universal_connected_sff(self.data.times)
        _ = plot(
            self.data.times,
            universal_sff,
            color=self.universal_sff_color,
            alpha=self.universal_sff_alpha,
            linewidth=self.universal_sff_width,
            zorder=self.universal_sff_zorder,
            label=self.universal_sff_legend,
        )

        self.finish_plot(path=path)
