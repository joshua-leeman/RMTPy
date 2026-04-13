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
class UnfoldedSpectralHistogramLegend(PlotLegend):
    loc: str = "upper right"
    bbox: tuple[float, float] = (0.94, 0.95)


@dataclass(repr=False, eq=False, kw_only=True)
class UnfoldedSpectralHistogramAxes(PlotAxes):
    xticks: tuple[float, ...] = (-0.5, 0.0, 0.5)  # units of dimension
    xticks_minor: tuple[float, ...] = (-0.25, 0.25)
    xlabel: str = r"$\xi$"
    xtick_labels: tuple[str, ...] = (
        r"$-\frac{D}{2}$",
        r"$0$",
        r"$\frac{D}{2}$",
    )

    yticks: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5)  # units of dimension^{-1}
    yticks_minor: tuple[float, ...] = (0.25, 0.75, 1.25, 1.75)
    ylabel: str = r"$\ensavg{\rho(\xi)}$"
    ytick_labels: tuple[str, ...] = (
        r"$0$",
        r"$\frac{1}{2 D}$",
        r"$\frac{1}{D}$",
        r"$\frac{3}{2 D}$",
    )


@dataclass(repr=False, eq=False, kw_only=True)
class UnfoldedSpectralHistogramPlot(Plot):
    data: Histogram
    axes: UnfoldedSpectralHistogramAxes = field(
        default_factory=UnfoldedSpectralHistogramAxes
    )
    num_points: int = 1000

    xlim: tuple[float, float] = (-0.6, 0.6)  # units of dimension
    ylim: tuple[float, float] = (0.0, 1.625)  # units of dimension^{-1}

    histogram_zorder: int = 1
    histogram_alpha: float = 0.5
    histogram_color: str = "RoyalBlue"
    histogram_legend: str = "simulation"

    pdf_zorder: int = 2
    pdf_width: float = 2.0
    pdf_alpha: float = 1.0
    pdf_color: str = "Black"
    pdf_legend: str = "theory"

    legend_labels: tuple[str, str] = (histogram_legend, pdf_legend)
    legend_handles: tuple[Patch, Line2D] = (
        Patch(color=histogram_color, alpha=histogram_alpha),
        Line2D([0], [0], color=pdf_color, linewidth=pdf_width),
    )

    def set_derived_attributes(self) -> None:
        try:
            ensemble_meta = self.data.metadata["simulation"]["args"]["ensemble"]
        except KeyError:
            raise ValueError("Ensemble metadata not found.")
        except TypeError:
            raise ValueError("Metadata is not properly structured.")

        self.ensemble: ManyBodyEnsemble = rmtpy_converter.structure(
            ensemble_meta, ManyBodyEnsemble
        )
        dimension: int = self.ensemble.dimension

        self.legend: UnfoldedSpectralHistogramLegend = UnfoldedSpectralHistogramLegend(
            handles=self.legend_handles, labels=self.legend_labels
        )
        if self.legend.title is None:
            self.legend.title = self.ensemble.to_latex + "\nunfolded"

        self.xlim = tuple(x * dimension for x in self.xlim)
        self.ylim = tuple(y / dimension for y in self.ylim)

        axes: UnfoldedSpectralHistogramAxes = self.axes
        axes.xticks = tuple(xtick * dimension for xtick in axes.xticks)
        axes.yticks = tuple(ytick / dimension for ytick in axes.yticks)
        axes.xticks_minor = tuple(xtick * dimension for xtick in axes.xticks_minor)
        axes.yticks_minor = tuple(ytick / dimension for ytick in axes.yticks_minor)

    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        histogram_bins = self.data.bins
        histogram = self.data.histogram

        self.create_figure()

        self.ax.hist(
            histogram_bins[:-1],
            bins=histogram_bins,
            weights=histogram,
            color=self.histogram_color,
            alpha=self.histogram_alpha,
            zorder=self.histogram_zorder,
        )

        energies = np.linspace(self.xlim[0], self.xlim[1], self.num_points)

        dimension: int = self.ensemble.dimension
        unfolded_spectral_pdf = np.zeros(self.num_points)
        unfolded_spectral_pdf[np.abs(energies) < dimension / 2] = 1 / dimension

        self.ax.plot(
            energies,
            unfolded_spectral_pdf,
            color=self.pdf_color,
            alpha=self.pdf_alpha,
            linewidth=self.pdf_width,
            zorder=self.pdf_zorder,
        )

        self.finish_plot(path=path)
