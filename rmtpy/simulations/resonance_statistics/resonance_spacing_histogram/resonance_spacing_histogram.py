from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..._histogram import Histogram
from ..._plot import PlotAxes, PlotLegend, Plot
from ....ensembles import ManyBodyEnsemble
from ....utils import rmtpy_converter


@dataclass(repr=False, eq=False, kw_only=True)
class ResonanceSpacingHistogramLegend(PlotLegend):
    loc: str = "upper right"
    bbox: tuple[float, float] = (0.94, 0.95)


@dataclass(repr=False, eq=False, kw_only=True)
class ResonanceSpacingHistogramAxes(PlotAxes):
    xticks: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0)  # units of mean spacing
    xticks_minor: tuple[float, ...] = (0.5, 1.5, 2.5, 3.5)
    xlabel: str = r"$\Delta E$"
    xtick_labels: tuple[str, ...] = (
        r"$0$",
        r"$d$",
        r"$2d$",
        r"$3d$",
        r"$4d$",
    )

    yticks: tuple[float, ...] = (0.5, 1.0)
    yticks_minor: tuple[float, ...] = (0.25, 0.75)
    ylabel: str = r"$\ensavg{f(\Delta E)}$"
    ytick_labels: tuple[str, ...] = (
        r"$\frac{1}{2}d^{-1}$",
        r"$d^{-1}$",
    )


@dataclass(repr=False, eq=False, kw_only=True)
class ResonanceSpacingHistogramPlot(Plot):
    data: Histogram
    axes: ResonanceSpacingHistogramAxes = field(
        default_factory=ResonanceSpacingHistogramAxes
    )
    num_points: int = 1000

    xlim: tuple[float, float] = (0.0, 4.0)  # units of mean spacing
    ylim: tuple[float, float] = (0.0, 1.2)

    histogram_zorder: int = 1
    histogram_alpha: float = 0.5
    histogram_color: str = "Orange"
    histogram_legend: str = "simulation"

    surmise_zorder: int = 2
    surmise_width: float = 2.0
    surmise_alpha: float = 1.0
    surmise_color: str = "Black"
    surmise_legend: str = "surmise"

    legend_labels: tuple[str, str] = (histogram_legend, surmise_legend)
    legend_handles: tuple[Patch, Line2D] = (
        Patch(color=histogram_color, alpha=histogram_alpha),
        Line2D([0], [0], color=surmise_color, linewidth=surmise_width),
    )

    def set_derived_attributes(self) -> None:
        try:
            ensemble_meta: dict = self.data.metadata["simulation"]["args"]["compound"][
                "args"
            ]["ensemble"]
        except KeyError:
            raise ValueError("Ensemble metadata not found.")
        except TypeError:
            raise ValueError("Metadata is not properly structured.")
        self.ensemble: ManyBodyEnsemble = rmtpy_converter.structure(
            ensemble_meta, ManyBodyEnsemble
        )

        self.legend: ResonanceSpacingHistogramLegend = ResonanceSpacingHistogramLegend(
            handles=self.legend_handles, labels=self.legend_labels
        )
        if self.legend.title is None:
            self.legend.title = self.ensemble.to_latex

        if self.ensemble.universality_class is not None:
            self.surmise_legend = f"{self.ensemble.universality_class} surmise"
            self.legend_labels = (self.histogram_legend, self.surmise_legend)

        mean_spacing: float = self.data.metadata["global_mean_spacing"]

        self.xlim = tuple(x * mean_spacing for x in self.xlim)
        self.ylim = tuple(y / mean_spacing for y in self.ylim)

        axes: ResonanceSpacingHistogramAxes = self.axes
        axes.xticks = tuple(xtick * mean_spacing for xtick in axes.xticks)
        axes.yticks = tuple(ytick / mean_spacing for ytick in axes.yticks)
        axes.xticks_minor = tuple(xtick * mean_spacing for xtick in axes.xticks_minor)
        axes.yticks_minor = tuple(ytick / mean_spacing for ytick in axes.yticks_minor)

    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        self.create_figure()

        self.ax.hist(
            self.data.bins[:-1],
            bins=self.data.bins,
            weights=self.data.histogram,
            color=self.histogram_color,
            alpha=self.histogram_alpha,
            zorder=self.histogram_zorder,
        )

        spacings: np.ndarray = np.linspace(0, self.xlim[1], self.num_points)

        mean_spacing: float = self.data.metadata["global_mean_spacing"]
        surmise: np.ndarray = self.ensemble.wigner_surmise(spacings / mean_spacing)
        surmise /= mean_spacing

        self.ax.plot(
            spacings,
            surmise,
            color=self.surmise_color,
            linewidth=self.surmise_width,
            alpha=self.surmise_alpha,
            zorder=self.surmise_zorder,
        )

        self.finish_plot(path=path)
