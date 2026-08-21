from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullLocator
from scipy.special import jn_zeros

import rmtpy.compounds

from ...plot import Plot, PlotAxes, PlotLegend
from ...spectral_statistics.spectral_form_factors import FormFactorsData


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class ResonanceFormFactorsAxes(PlotAxes):
    xticks: tuple[float, ...] = (0.0, 0.5, 1.0)  # log scale base dimension
    # t_0 = j_\text{\tiny 1,1} / J
    xlabel: str = r"$u = t / t_0$"
    xtick_labels: tuple[str, ...] = (
        r"$N_\text{m}^{-1}$",
        r"$D^{1/2} N_\text{m}^{-1}$",
        r"$D N_\text{m}^{-1}$",
    )

    yticks: tuple[float, ...] = (-2, -1, 0)  # log scale base dimension
    ytick_labels: tuple[str, ...] = (
        r"$D^{-2}$",
        r"$D^{-1}$",
        r"$1$",
    )


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class ResonanceFormFactorsPlot(Plot):
    data: FormFactorsData
    axes: ResonanceFormFactorsAxes = dataclasses.field(
        default_factory=ResonanceFormFactorsAxes
    )
    num_points: int = 1000

    xlim: tuple[float, float] = (-0.5, 1.5)  # log scale base dimension
    ylim: tuple[float, float] = (-2.2, 0.2)

    # thouless_marker: str = "*"
    # thouless_size: int = 12
    # thouless_color: str = "Black"
    # thouless_alpha: float = 1.0
    # thouless_zorder: int = 3
    # thouless_style: str = "None"
    # thouless_legend: str = r"$t_\text{\tiny Th}$"

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

    legend_labels: tuple[str, str] = (sff_legend, csff_legend)  # , thou_legend)
    legend_handles: tuple[Line2D, Line2D] = (
        Line2D([0], [0], color=sff_color, alpha=sff_alpha, linewidth=sff_width),
        Line2D([0], [0], color=csff_color, alpha=csff_alpha, linewidth=csff_width),
        # Line2D(
        #     [0],
        #     [0],
        #     marker=thouless_marker,
        #     color=thouless_color,
        #     linestyle=thouless_style,
        # ),
    )

    def set_derived_attributes(self) -> None:
        self.compound: rmtpy.compounds.Compound = self.structure_simulation_arg(
            "compound", rmtpy.compounds.Compound
        )
        mean_coupling_squared = np.mean(self.compound.coupling_strengths**2)
        ensemble = self.compound.ensemble

        self.legend = PlotLegend(
            handles=self.legend_handles,
            labels=self.legend_labels,
            loc="upper right",
            bbox=(0.925, 0.95),
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

        j_1_1 = float(jn_zeros(1, 1)[0])
        self.scale_limits_and_ticks(
            x=lambda value: (
                self.compound.ensemble.dimension**value
                * (j_1_1 / self.compound.ensemble.spectral_radius)
            ),
            y=lambda value: self.compound.ensemble.dimension**value,
        )

    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        self.create_figure()

        self.ax.set_xscale("log", base=self.compound.ensemble.dimension)
        self.ax.set_yscale("log", base=self.compound.ensemble.dimension)

        self.ax.xaxis.set_major_locator(
            LogLocator(
                base=self.compound.ensemble.dimension, numticks=len(self.axes.xticks)
            )
        )
        self.ax.xaxis.set_minor_locator(NullLocator())
        self.ax.yaxis.set_major_locator(
            LogLocator(
                base=self.compound.ensemble.dimension, numticks=len(self.axes.yticks)
            )
        )
        self.ax.yaxis.set_minor_locator(NullLocator())

        self.ax.plot(
            self.data.times,
            self.data.form_factor,
            color=self.sff_color,
            alpha=self.sff_alpha,
            linewidth=self.sff_width,
            zorder=self.sff_zorder,
            label=self.sff_legend,
        )

        self.ax.plot(
            self.data.times,
            self.data.connected_form_factor,
            color=self.csff_color,
            alpha=self.csff_alpha,
            linewidth=self.csff_width,
            zorder=self.csff_zorder,
            label=self.csff_legend,
        )

        self.finish_plot(path=path)


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class UnfoldedResonanceFormFactorsAxes(PlotAxes):
    xticks: tuple[float, ...] = (-1.0, -0.5, 0.0)  # log scale base dimension
    xlabel: str = r"$\upsilon = \tau / \tau_\text{\tiny H}$"
    xtick_labels: tuple[str, ...] = (
        r"$D^{-1}$",
        r"$D^{-1/2}$",
        r"$1$",
    )

    yticks: tuple[float, ...] = (-2, -1, 0)  # log scale base dimension
    ytick_labels: tuple[str, ...] = (
        r"$D^{-2}$",
        r"$D^{-1}$",
        r"$1$",
    )


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class UnfoldedResonanceFormFactorsPlot(Plot):
    data: FormFactorsData
    axes: UnfoldedResonanceFormFactorsAxes = dataclasses.field(
        default_factory=UnfoldedResonanceFormFactorsAxes
    )
    num_points: int = 1000

    xlim: tuple[float, float] = (-1.5, 0.5)  # log scale base dimension
    ylim: tuple[float, float] = (-2.2, 0.2)

    # thouless_marker: str = "*"
    # thouless_size: int = 12
    # thouless_color: str = "Black"
    # thouless_alpha: float = 1.0
    # thouless_zorder: int = 3
    # thouless_style: str = "None"
    # thouless_legend: str = r"$t_\text{\tiny Th}$"

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
        self.compound: rmtpy.compounds.Compound = self.structure_simulation_arg(
            "compound", rmtpy.compounds.Compound
        )
        mean_coupling_squared = np.mean(self.compound.coupling_strengths**2)
        ensemble = self.compound.ensemble

        if self.compound.ensemble.universality_class is not None:
            self.universal_sff_legend = rf"$K^{{\text{{\tiny {self.compound.ensemble.universality_class}}}}}_{{\text{{\tiny conn}}}}(\upsilon)$"
            self.legend_labels = (
                self.sff_legend,
                self.csff_legend,
                self.universal_sff_legend,
            )

        self.legend = PlotLegend(
            handles=self.legend_handles,
            labels=self.legend_labels,
            loc="upper right",
            bbox=(0.925, 0.95),
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
            x=lambda value: self.compound.ensemble.dimension**value * 2 * np.pi,
            y=lambda value: self.compound.ensemble.dimension**value,
        )

    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        self.create_figure()

        self.ax.set_xscale("log", base=self.compound.ensemble.dimension)
        self.ax.set_yscale("log", base=self.compound.ensemble.dimension)

        self.ax.xaxis.set_major_locator(
            LogLocator(
                base=self.compound.ensemble.dimension, numticks=len(self.axes.xticks)
            )
        )
        self.ax.xaxis.set_minor_locator(NullLocator())
        self.ax.yaxis.set_major_locator(
            LogLocator(
                base=self.compound.ensemble.dimension, numticks=len(self.axes.yticks)
            )
        )
        self.ax.yaxis.set_minor_locator(NullLocator())

        self.ax.plot(
            self.data.times,
            self.data.form_factor,
            color=self.sff_color,
            alpha=self.sff_alpha,
            linewidth=self.sff_width,
            zorder=self.sff_zorder,
            label=self.sff_legend,
        )

        self.ax.plot(
            self.data.times,
            self.data.connected_form_factor,
            color=self.csff_color,
            alpha=self.csff_alpha,
            linewidth=self.csff_width,
            zorder=self.csff_zorder,
            label=self.csff_legend,
        )

        universal_sff = self.compound.ensemble.universal_connected_sff(self.data.times)

        self.ax.plot(
            self.data.times,
            universal_sff,
            color=self.universal_sff_color,
            alpha=self.universal_sff_alpha,
            linewidth=self.universal_sff_width,
            zorder=self.universal_sff_zorder,
            label=self.universal_sff_legend,
        )

        self.finish_plot(path=path)
