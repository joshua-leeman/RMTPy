import dataclasses
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from rmtpy.ensembles import (
    ManyBodyEnsemble,
    PoissonEnsemble,
    SachdevYeKitaevEnsemble,
)

from ...histogram import Histogram
from ...plot import Plot, PlotAxes, PlotLegend


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class SpectralHistogramLegend(PlotLegend):
    loc: str = "upper right"
    bbox: tuple[float, float] = (0.94, 0.95)


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class SpectralHistogramAxes(PlotAxes):
    xticks: tuple[float, ...] = (-1.0, 0.0, 1.0)  # units of energy_0
    xticks_minor: tuple[float, ...] = (-0.5, 0.5)
    xlabel: str = r"$E$"
    xtick_labels: tuple[str, ...] = (
        r"$-E_0$",
        r"$0$",
        r"$E_0$",
    )

    yticks: tuple[float, ...] = (0.0, 1.0, 2.0)  # units of 1 / (pi * energy_0)
    yticks_minor: tuple[float, ...] = (0.0, 1.0, 2.0)
    ylabel: str = r"$\ensavg{\rho(E)}$"
    ytick_labels: tuple[str, ...] = (
        r"$0$",
        r"$\frac{1}{\pi E_0}$",
        r"$\frac{2}{\pi E_0}$",
    )

    poisson_yticks: tuple[float, ...] = (
        0.0,
        0.25 * np.pi,
        0.5 * np.pi,
        0.75 * np.pi,
    )  # units of 1 / (pi * energy_0)
    poisson_yticks_minor: tuple[float, ...] = (
        0.125 * np.pi,
        0.375 * np.pi,
        0.625 * np.pi,
        0.875 * np.pi,
    )
    poisson_ytick_labels: tuple[str, ...] = (
        r"$0$",
        r"$\frac{1}{4E_0}$",
        r"$\frac{1}{2E_0}$",
        r"$\frac{3}{4E_0}$",
    )

    syk2_yticks: tuple[float, ...] = tuple(range(6))  # units of 1 / (pi * energy_0)
    syk2_yticks_minor: tuple[float, ...] = tuple(x + 0.5 for x in range(6))
    syk2_ytick_labels: tuple[str, ...] = (
        r"$0$",
        r"$\frac{1}{\pi E_0}$",
        r"$\frac{2}{\pi E_0}$",
        r"$\frac{3}{\pi E_0}$",
        r"$\frac{4}{\pi E_0}$",
        r"$\frac{5}{\pi E_0}$",
    )

    syk4_yticks: tuple[float, ...] = tuple(range(3))  # units of 1 / (pi * energy_0)
    syk4_yticks_minor: tuple[float, ...] = tuple(x + 0.5 for x in range(3))
    syk4_ytick_labels: tuple[str, ...] = (
        r"$0$",
        r"$\frac{1}{\pi E_0}$",
        r"$\frac{2}{\pi E_0}$",
    )


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class SpectralHistogramPlot(Plot):
    data: Histogram
    axes: SpectralHistogramAxes = dataclasses.field(
        default_factory=SpectralHistogramAxes
    )
    num_points: int = 1000

    xlim: tuple[float, float] = (-1.2, 1.2)  # units of energy_0
    ylim: tuple[float, float] = (0.0, 2.6)  # units of 1 / (pi * energy_0)

    poisson_ylim: tuple[float, float] = (0.0, 1.25)  # units of 1 / (pi * energy_0)
    syk2_ylim: tuple[float, float] = (0.0, 4.0)
    syk4_ylim: tuple[float, float] = (0.0, 2.5)

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
        self.ensemble: ManyBodyEnsemble = self.structure_simulation_arg(
            "ensemble", ManyBodyEnsemble
        )
        energy_0: float = self.ensemble.spectral_radius

        self.legend: SpectralHistogramLegend = SpectralHistogramLegend(
            handles=self.legend_handles, labels=self.legend_labels
        )
        if self.legend.title is None:
            self.legend.title = self.ensemble.to_latex

        axes: SpectralHistogramAxes = self.axes
        if isinstance(self.ensemble, PoissonEnsemble):
            self.ylim = self.poisson_ylim

            axes.ytick_labels = axes.poisson_ytick_labels
            axes.yticks = axes.poisson_yticks
            axes.yticks_minor = axes.poisson_yticks_minor

        elif isinstance(self.ensemble, SachdevYeKitaevEnsemble):
            if self.ensemble.q == 2:
                self.ylim = self.syk2_ylim

                axes.ytick_labels = axes.syk2_ytick_labels
                axes.yticks = axes.syk2_yticks
                axes.yticks_minor = axes.syk2_yticks_minor

            elif self.ensemble.q == 4:
                self.ylim = self.syk4_ylim

                axes.ytick_labels = axes.syk4_ytick_labels
                axes.yticks = axes.syk4_yticks
                axes.yticks_minor = axes.syk4_yticks_minor

        self.scale_limits_and_ticks(
            x=lambda value: value * energy_0,
            y=lambda value: value / np.pi / energy_0,
        )

    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        self.create_figure()

        self.draw_histogram(
            color=self.histogram_color,
            alpha=self.histogram_alpha,
            zorder=self.histogram_zorder,
        )

        energies: np.ndarray = np.linspace(*self.xlim, self.num_points)
        spectral_pdf: np.ndarray = self.ensemble.spectral_density.average_pdf(energies)

        self.ax.plot(
            energies,
            spectral_pdf,
            color=self.pdf_color,
            alpha=self.pdf_alpha,
            linewidth=self.pdf_width,
            zorder=self.pdf_zorder,
        )

        self.finish_plot(path=path)


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class UnfoldedSpectralHistogramLegend(PlotLegend):
    loc: str = "upper right"
    bbox: tuple[float, float] = (0.94, 0.95)


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
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


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class UnfoldedSpectralHistogramPlot(Plot):
    data: Histogram
    axes: UnfoldedSpectralHistogramAxes = dataclasses.field(
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
        self.ensemble: ManyBodyEnsemble = self.structure_simulation_arg(
            "ensemble", ManyBodyEnsemble
        )
        dimension: int = self.ensemble.dimension

        self.legend: UnfoldedSpectralHistogramLegend = UnfoldedSpectralHistogramLegend(
            handles=self.legend_handles, labels=self.legend_labels
        )
        if self.legend.title is None:
            self.legend.title = self.ensemble.to_latex + "\nunfolded"

        self.scale_limits_and_ticks(
            x=lambda value: value * dimension,
            y=lambda value: value / dimension,
        )

    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        self.create_figure()

        self.draw_histogram(
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
