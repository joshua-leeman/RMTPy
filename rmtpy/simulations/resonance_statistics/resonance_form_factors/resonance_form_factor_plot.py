from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullLocator
from scipy.special import jn_zeros

from .resonance_form_factor_data import FormFactorsData
from ..._plot import PlotLegend, PlotAxes, Plot
from ....compounds import Compound
from ....utils import rmtpy_converter


@dataclass(repr=False, eq=False, kw_only=True)
class ResonanceFormFactorsLegend(PlotLegend):
    loc: str = "upper right"
    bbox: tuple[float, float] = (0.735, 0.9)


@dataclass(repr=False, eq=False, kw_only=True)
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


@dataclass(repr=False, eq=False, kw_only=True)
class ResonanceFormFactorsPlot(Plot):
    data: FormFactorsData
    axes: ResonanceFormFactorsAxes = field(default_factory=ResonanceFormFactorsAxes)
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
        try:
            compound_meta: dict = self.data.metadata["simulation"]["args"]["compound"]
        except KeyError:
            raise ValueError("Compound metadata not found.")
        except TypeError:
            raise ValueError("Metadata is not properly structured.")
        self.compound: Compound = rmtpy_converter.structure(compound_meta, Compound)
        energy_0: float = self.compound.ensemble.ground_state_energy
        dimension: int = self.compound.ensemble.dimension

        self.legend: ResonanceFormFactorsLegend = ResonanceFormFactorsLegend(
            handles=self.legend_handles, labels=self.legend_labels
        )

        if self.legend.title is None:
            self.legend.title = self.compound.to_latex

        j_1_1: float = float(jn_zeros(1, 1)[0])
        self.xlim = tuple(dimension**x * j_1_1 / energy_0 for x in self.xlim)
        self.ylim = tuple(dimension**y for y in self.ylim)

        axes: ResonanceFormFactorsAxes = self.axes
        axes.xticks = tuple(dimension**x * j_1_1 / energy_0 for x in axes.xticks)
        axes.yticks = tuple(dimension**y for y in axes.yticks)

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
