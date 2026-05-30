from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullLocator
from scipy.special import jn_zeros

from ....compounds import Compound
from ....ensembles import ManyBodyEnsemble
from ...plot import Plot, PlotAxes, PlotLegend
from ...spectral_statistics.spectral_form_factors import FormFactorsData


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class ResonanceFormFactorsLegend(PlotLegend):
    loc: str = "upper right"
    bbox: tuple[float, float] = (0.735, 0.9)


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class ResonanceFormFactorsAxes(PlotAxes):
    xticks: tuple[float, ...] = (0.0, 0.5, 1.0)  # log scale base dimension
    xlabel: str = r"$N_\textrm{\tiny m} Jt / j_\textrm{\tiny 1,1}$"
    xtick_labels: tuple[str, ...] = (
        r"$1$",
        r"$D^{1/2}$",
        r"$D$",
    )

    yticks: tuple[float, ...] = (-2, -1, 0)  # log scale base dimension
    ylabel: str = r"$K(t)$"
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
    # thouless_legend: str = r"$t_\textrm{\tiny Th}$"

    sff_zorder: int = 2
    sff_width: float = 0.5
    sff_alpha: float = 1.0
    sff_color: str = "Blue"
    sff_legend: str = "SFF"

    csff_zorder: int = 2
    csff_width: float = 0.5
    csff_alpha: float = 1.0
    csff_color: str = "Red"
    csff_legend: str = "cSFF"

    grid_zorder: int = 0
    grid_width: float = rcParams["grid.linewidth"]
    grid_alpha: float = 1.0
    grid_color: str = rcParams["grid.color"]
    grid_linestyle: str = "dotted"

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
        self.compound: Compound = self.structure_simulation_arg("compound", Compound)
        energy_0: float = self.compound.ensemble.spectral_radius
        dimension: int = self.compound.ensemble.dimension

        self.legend: ResonanceFormFactorsLegend = ResonanceFormFactorsLegend(
            handles=self.legend_handles, labels=self.legend_labels
        )

        if self.legend.title is None:
            self.legend.title = self.compound.to_latex

        j_1_1: float = float(jn_zeros(1, 1)[0])
        self.scale_limits_and_ticks(
            x=lambda value: dimension**value * j_1_1 / energy_0,
            y=lambda value: dimension**value,
        )

    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        self.create_figure()

        dimension: int = self.compound.ensemble.dimension

        self.ax.set_xscale("log", base=dimension)
        self.ax.set_yscale("log", base=dimension)

        self.ax.xaxis.set_major_locator(
            LogLocator(base=dimension, numticks=len(self.axes.xticks))
        )
        self.ax.xaxis.set_minor_locator(NullLocator())
        self.ax.yaxis.set_major_locator(
            LogLocator(base=dimension, numticks=len(self.axes.yticks))
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

        self.ax.vlines(
            self.axes.xticks,
            ymin=self.ylim[0],
            ymax=self.ylim[1],
            colors=self.grid_color,
            linestyles=self.grid_linestyle,
            linewidth=self.grid_width,
            alpha=self.grid_alpha,
            zorder=self.grid_zorder,
        )

        self.finish_plot(path=path)


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class UnfoldedResonanceFormFactorsLegend(PlotLegend):
    loc: str = "upper right"
    bbox: tuple[float, float] = (0.76, 0.96)


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class UnfoldedResonanceFormFactorsAxes(PlotAxes):
    xticks: tuple[float, ...] = (-1.0, -0.5, 0.0)  # log scale base dimension
    xlabel: str = r"$\tau / \tau_\textrm{\tiny H}$"
    xtick_labels: tuple[str, ...] = (
        r"$D^{-1}$",
        r"$D^{-1/2}$",
        r"$1$",
    )

    yticks: tuple[float, ...] = (-2, -1, 0)  # log scale base dimension
    ylabel: str = r"$K(\tau)$"
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
    # thouless_legend: str = r"$t_\textrm{\tiny Th}$"

    sff_zorder: int = 2
    sff_width: float = 0.5
    sff_alpha: float = 1.0
    sff_color: str = "Blue"
    sff_legend: str = "SFF"

    csff_zorder: int = 2
    csff_width: float = 0.5
    csff_alpha: float = 1.0
    csff_color: str = "Red"
    csff_legend: str = "cSFF"

    universal_csff_zorder: int = 2
    universal_csff_width: float = 0.5
    universal_csff_alpha: float = 1.0
    universal_csff_color: str = "Black"
    universal_csff_legend: str = "universal"

    grid_zorder: int = 0
    grid_width: float = rcParams["grid.linewidth"]
    grid_alpha: float = 1.0
    grid_color: str = rcParams["grid.color"]
    grid_linestyle: str = "dotted"

    legend_labels: tuple[str, str, str] = (
        sff_legend,
        csff_legend,
        universal_csff_legend,
    )
    legend_handles: tuple[Line2D, Line2D, Line2D] = (
        Line2D([0], [0], color=sff_color, alpha=sff_alpha, linewidth=sff_width),
        Line2D([0], [0], color=csff_color, alpha=csff_alpha, linewidth=csff_width),
        Line2D(
            [0],
            [0],
            color=universal_csff_color,
            alpha=universal_csff_alpha,
            linewidth=universal_csff_width,
        ),
    )

    def set_derived_attributes(self) -> None:
        self.compound: Compound = self.structure_simulation_arg("compound", Compound)
        ensemble: ManyBodyEnsemble = self.compound.ensemble
        dimension: int = ensemble.dimension

        if ensemble.universality_class is not None:
            self.universal_csff_legend = f"{ensemble.universality_class} limit"
            self.legend_labels = (
                self.sff_legend,
                self.csff_legend,
                self.universal_csff_legend,
            )

        self.legend: UnfoldedResonanceFormFactorsLegend = (
            UnfoldedResonanceFormFactorsLegend(
                handles=self.legend_handles, labels=self.legend_labels
            )
        )
        if self.legend.title is None:
            self.legend.title = self.compound.to_latex

        self.scale_limits_and_ticks(
            x=lambda value: dimension**value * 2 * np.pi,
            y=lambda value: dimension**value,
        )

    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        ensemble: ManyBodyEnsemble = self.compound.ensemble
        dimension: int = ensemble.dimension

        self.create_figure()

        self.ax.set_xscale("log", base=dimension)
        self.ax.set_yscale("log", base=dimension)

        self.ax.xaxis.set_major_locator(
            LogLocator(base=dimension, numticks=len(self.axes.xticks))
        )
        self.ax.xaxis.set_minor_locator(NullLocator())
        self.ax.yaxis.set_major_locator(
            LogLocator(base=dimension, numticks=len(self.axes.yticks))
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

        universal_csff = ensemble.universal_csff(self.data.times)

        self.ax.plot(
            self.data.times,
            universal_csff,
            color=self.universal_csff_color,
            alpha=self.universal_csff_alpha,
            linewidth=self.universal_csff_width,
            zorder=self.universal_csff_zorder,
            label=self.universal_csff_legend,
        )

        self.ax.vlines(
            self.axes.xticks,
            ymin=self.ylim[0],
            ymax=self.ylim[1],
            colors=self.grid_color,
            linestyles=self.grid_linestyle,
            linewidth=self.grid_width,
            alpha=self.grid_alpha,
            zorder=self.grid_zorder,
        )

        self.finish_plot(path=path)
