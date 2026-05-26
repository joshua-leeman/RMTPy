from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from matplotlib.patches import Patch

import rmtpy.conversion
import rmtpy.ensembles
from ...histogram import Histogram
from ...plot import Plot, PlotAxes, PlotLegend


@dataclass(repr=False, eq=False, kw_only=True)
class SpectralCoefficientHistogramLegend(PlotLegend):
    loc: str = "upper right"
    bbox: tuple[float, float] = (0.94, 0.95)


@dataclass(repr=False, eq=False, kw_only=True)
class SpectralCoefficientHistogramAxes(PlotAxes):
    xticks: tuple[float, ...] = (-0.2, -0.1, 0.0, 0.1, 0.2)  # units of energy_0
    xticks_minor: tuple[float, ...] = (-0.15, -0.05, 0.05, 0.15)
    xlabel: str = r"$c$"
    xtick_labels: tuple[str, ...] = (
        r"$-0.2$",
        r"$-0.1$",
        r"$0.0$",
        r"$+0.1$",
        r"$+0.2$",
    )

    yticks: tuple[float, ...] = tuple(range(0, 24, 4))  # units of 1 / (pi * energy_0)
    yticks_minor: tuple[float, ...] = tuple(range(2, 22, 4))
    ylabel: str = r"$\ensavg{\rho(c)}$"
    ytick_labels: tuple[str, ...] = (
        r"$0$",
        r"$4$",
        r"$8$",
        r"$12$",
        r"$16$",
        r"$20$",
    )


@dataclass(repr=False, eq=False, kw_only=True)
class SpectralCoefficientHistogramPlot(Plot):
    data: Histogram
    axes: SpectralCoefficientHistogramAxes = field(
        default_factory=SpectralCoefficientHistogramAxes
    )

    xlim: tuple[float, float] = (-0.2, 0.2)  # units of energy_0
    ylim: tuple[float, float] = (0.0, 20)  # units of 1 / (pi * energy_0)

    histogram_zorder: int = 1
    histogram_alpha: float = 0.5
    histogram_color: str = "RoyalBlue"
    histogram_legend: str = "simulation"

    legend_labels: tuple[str] = (histogram_legend,)
    legend_handles: tuple[Patch] = (
        Patch(color=histogram_color, alpha=histogram_alpha),
    )

    def set_derived_attributes(self) -> None:
        try:
            ensemble_meta: dict = self.data.metadata["simulation"]["args"]["ensemble"]
        except KeyError:
            raise ValueError("Ensemble metadata not found.")
        except TypeError:
            raise ValueError("Metadata is not properly structured.")

        self.ensemble: rmtpy.ensembles.ManyBodyEnsemble = (
            rmtpy.conversion.CONVERTER.structure(
                ensemble_meta, rmtpy.ensembles.ManyBodyEnsemble
            )
        )

        self.legend: SpectralCoefficientHistogramLegend = (
            SpectralCoefficientHistogramLegend(
                handles=self.legend_handles, labels=self.legend_labels
            )
        )
        if self.legend.title is None:
            self.legend.title = self.ensemble.to_latex

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

        self.finish_plot(path=path)
