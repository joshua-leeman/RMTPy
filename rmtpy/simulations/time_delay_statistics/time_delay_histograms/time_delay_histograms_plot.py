from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
from matplotlib import rcParams
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullLocator

from rmtpy.compounds import Compound

from ...histogram import Histogram
from ...plot import Plot, PlotAxes, PlotLegend

TIME_DELAY_HISTOGRAM_COLOR: str = "#7b2d26"


def format_energy_label(energy: float, energy_0: float) -> str:
    scaled_energy = energy / energy_0
    if np.isclose(scaled_energy, 0.0):
        return r"$E = 0$"

    return rf"$E = {scaled_energy:.3g}E_0$"


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class TimeDelayHistogramAxes(PlotAxes):
    xticks: tuple[float, ...] = (0.0, 0.5, 1.0)  # log scale base dimension
    xlabel: str = r"$N_\textrm{\tiny m} Jt / j_\textrm{\tiny 1,1}$"
    xtick_labels: tuple[str, ...] = (
        r"$1$",
        r"$D^{1/2}$",
        r"$D$",
    )

    ylabel: str = r"$\diff P / \diff t$"


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class TimeDelayHistogramLegend(PlotLegend):
    loc: str = "upper right"
    bbox: tuple[float, float] = (0.94, 0.95)


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class TimeDelayHistogramPlot(Plot):
    data: Histogram
    axes: TimeDelayHistogramAxes = dataclasses.field(
        default_factory=TimeDelayHistogramAxes
    )

    xlim: tuple[float, float] = (-0.5, 1.5)  # log scale base dimension

    histogram_zorder: int = 1
    histogram_alpha: float = 0.42
    histogram_color: str = TIME_DELAY_HISTOGRAM_COLOR

    grid_zorder: int = 0
    grid_width: float = rcParams["grid.linewidth"]
    grid_alpha: float = 1.0
    grid_color: str = rcParams["grid.color"]
    grid_linestyle: str = "dotted"

    pdf_zorder: int = 2
    pdf_width: float = 2.0
    pdf_alpha: float = 1.0
    pdf_color: str = "Black"
    pdf_legend: str = "theory"

    num_points: int = 1000

    def set_derived_attributes(self) -> None:
        self.compound = self.structure_simulation_arg("compound", Compound)

        energy_0 = self.compound.ensemble.spectral_radius
        dimension = self.compound.ensemble.dimension
        energy = self.data.metadata["energy"]

        self.legend = TimeDelayHistogramLegend(
            handles=(Patch(color=self.histogram_color, alpha=self.histogram_alpha),),
            labels=(format_energy_label(energy, energy_0),),
        )
        if self.legend.title is None:
            self.legend.title = self.compound.to_latex

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

        self.ax.vlines(
            self.axes.xticks,
            ymin=0.0,
            ymax=max(self.data.histogram) if np.any(self.data.histogram) else 1.0,
            colors=self.grid_color,
            linestyles=self.grid_linestyle,
            linewidth=self.grid_width,
            alpha=self.grid_alpha,
            zorder=self.grid_zorder,
        )

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
class UnfoldedTimeDelayHistogramAxes(PlotAxes):
    xticks: tuple[float, ...] = (-1.0, -0.5, 0.0)  # log scale base dimension
    xlabel: str = r"$\tau / \tau_\textrm{\tiny H}$"
    xtick_labels: tuple[str, ...] = (
        r"$D^{-1}$",
        r"$D^{-1/2}$",
        r"$1$",
    )

    ylabel: str = r"$\diff P / \diff \tau$"


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class UnfoldedTimeDelayHistogramPlot(TimeDelayHistogramPlot):
    axes: UnfoldedTimeDelayHistogramAxes = dataclasses.field(
        default_factory=UnfoldedTimeDelayHistogramAxes
    )

    xlim: tuple[float, float] = (-1.5, 0.5)  # log scale base dimension

    def set_derived_attributes(self) -> None:
        super().set_derived_attributes()

        if self.legend.title is None:
            self.legend.title = self.compound.to_latex
