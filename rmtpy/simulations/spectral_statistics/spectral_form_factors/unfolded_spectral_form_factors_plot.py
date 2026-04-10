from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullLocator

from .spectral_form_factors_data import FormFactorsData
from ..._plot import PlotLegend, PlotAxes, Plot
from ....ensembles import ManyBodyEnsemble
from ....utils import rmtpy_converter


@dataclass(repr=False, eq=False, kw_only=True)
class UnfoldedFormFactorsLegend(PlotLegend):
    loc: str = "upper right"
    bbox: tuple[float, float] = (0.76, 0.96)


@dataclass(repr=False, eq=False, kw_only=True)
class UnfoldedFormFactorsAxes(PlotAxes):
    xticks: tuple[float, ...] = (-1.0, -0.5, 0.0)  # factor of 2 pi
    xlabel: str = r"$\tau / \tau_\textrm{\tiny H}$"
    xtick_labels: tuple[str, ...] = (
        r"$D^{-1}$",
        r"$D^{-1/2}$",
        r"$1$",
    )

    yticks: tuple[float, ...] = (-2, -1, 0)  # log scale base dim
    ylabel: str = r"$K(\tau)$"
    ytick_labels: tuple[str, ...] = (
        r"$D^{-2}$",
        r"$D^{-1}$",
        r"$1$",
    )


@dataclass(repr=False, eq=False, kw_only=True)
class UnfoldedFormFactorsPlot(Plot):
    data: FormFactorsData
    axes: UnfoldedFormFactorsAxes = field(default_factory=UnfoldedFormFactorsAxes)
    num_points: int = 1000

    xlim: tuple[float, float] = (-1.5, 0.5)  # log scale base dim
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
        try:
            ensemble_meta: dict = self.data.metadata["simulation"]["args"]["ensemble"]
        except KeyError:
            raise ValueError("Ensemble metadata not found.")
        except TypeError:
            raise ValueError("Metadata is not properly structured.")
        self.ensemble: ManyBodyEnsemble = rmtpy_converter.structure(
            ensemble_meta, ManyBodyEnsemble
        )
        dimension: int = self.ensemble.dimension

        if self.ensemble.universality_class is not None:
            self.universal_csff_legend = f"{self.ensemble.universality_class} limit"
            self.legend_labels = (
                self.sff_legend,
                self.csff_legend,
                self.universal_csff_legend,
            )

        self.legend = UnfoldedFormFactorsLegend(
            handles=self.legend_handles, labels=self.legend_labels
        )
        if self.legend.title is None:
            self.legend.title = self.ensemble.to_latex

        self.xlim = tuple(dimension**x * 2 * np.pi for x in self.xlim)
        self.ylim = tuple(dimension**y for y in self.ylim)

        axes: UnfoldedFormFactorsAxes = self.axes
        axes.xticks = tuple(dimension**x * 2 * np.pi for x in axes.xticks)
        axes.yticks = tuple(dimension**y for y in axes.yticks)

    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        self.create_figure()

        self.ax.set_xscale("log", base=self.ensemble.dimension)
        self.ax.set_yscale("log", base=self.ensemble.dimension)

        self.ax.xaxis.set_major_locator(
            LogLocator(base=self.ensemble.dimension, numticks=len(self.axes.xticks))
        )
        self.ax.xaxis.set_minor_locator(NullLocator())
        self.ax.yaxis.set_major_locator(
            LogLocator(base=self.ensemble.dimension, numticks=len(self.axes.yticks))
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

        universal_csff = self.ensemble.universal_csff(self.data.times)

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
