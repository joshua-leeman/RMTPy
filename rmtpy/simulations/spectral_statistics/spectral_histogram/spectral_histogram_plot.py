import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import cast, override

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ....ensembles import (
    ManyBodyEnsemble,
    PoissonEnsemble,
    SachdevYeKitaevEnsemble,
)
from ...base_data import Data
from ...base_plot import Plot, PlotAxes, PlotLegend

_POLYNOMIAL_WEIGHT_LEGEND: str = "polynomial weight"


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class SpectralHistogramAxes(PlotAxes):
    xticks: tuple[float, ...] = (-1.0, 0.0, 1.0)  # units of energy_0
    xticks_minor: tuple[float, ...] = (-0.5, 0.5)
    xlabel: str = r"$E / E_0$"
    xtick_labels: tuple[str, ...] = (
        r"$-1.0$",
        r"$0.0$",
        r"$+1.0$",
    )

    yticks: tuple[float, ...] = (0.0, 1.0, 2.0)  # units of 1 / (pi * energy_0)
    yticks_minor: tuple[float, ...] = (0.0, 1.0, 2.0)
    ylabel: str = r"$\pi E_0 \ensavg{\rho(E)}$"
    ytick_labels: tuple[str, ...] = (
        r"$0.0$",
        r"$1.0$",
        r"$2.0$",
    )

    pois_yticks: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0)  # units of 1 / (pi * energy_0)
    pois_yticks_minor: tuple[float, ...] = (0.5, 1.5, 2.5)
    pois_ytick_labels: tuple[str, ...] = (
        r"$0.0$",
        r"$1.0$",
        r"$2.0$",
        r"$3.0$",
    )

    syk2_yticks: tuple[float, ...] = tuple(range(6))  # units of 1 / (pi * energy_0)
    syk2_yticks_minor: tuple[float, ...] = tuple(x + 0.5 for x in range(6))
    syk2_ytick_labels: tuple[str, ...] = (
        r"$0.0$",
        r"$1.0$",
        r"$2.0$",
        r"$3.0$",
        r"$4.0$",
        r"$5.0$",
    )

    syk4_yticks: tuple[float, ...] = tuple(range(3))  # units of 1 / (pi * energy_0)
    syk4_yticks_minor: tuple[float, ...] = tuple(x + 0.5 for x in range(3))
    syk4_ytick_labels: tuple[str, ...] = (
        r"$0.0$",
        r"$1.0$",
        r"$2.0$",
    )


@dataclasses.dataclass(kw_only=True, eq=False, weakref_slot=False)
class SpectralHistogramPlot(Plot):
    data: Data
    axes: PlotAxes = dataclasses.field(default_factory=SpectralHistogramAxes)

    xlim: tuple[float, float] = (-1.2, 1.2)  # units of energy_0
    ylim: tuple[float, float] = (0.0, 2.6)  # units of 1 / (pi * energy_0)

    pois_ylim: tuple[float, float] = (0.0, 1.25)  # units of 1 / (pi * energy_0)
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

    num_points: int = 1000

    legend_labels: tuple[str, str] = (histogram_legend, pdf_legend)
    legend_handles: tuple[Patch, Line2D] = (
        Patch(color=histogram_color, alpha=histogram_alpha),
        Line2D([0], [0], color=pdf_color, linewidth=pdf_width),
    )

    def set_derived_attributes(self) -> None:
        self.ensemble: ManyBodyEnsemble = self.store_manifest_arg(
            "ensemble", ManyBodyEnsemble
        )

        self.legend: PlotLegend = PlotLegend(
            handles=self.legend_handles,
            labels=self.legend_labels,
            loc="upper right",
            bbox=(0.99, 0.95),
        )
        if not self.legend.title:
            self.legend.title = self.ensemble.to_latex

        axes = cast(SpectralHistogramAxes, self.axes)
        if isinstance(self.ensemble, PoissonEnsemble):
            self.ylim = self.pois_ylim

            axes.ytick_labels = axes.pois_ytick_labels
            axes.yticks = axes.pois_yticks
            axes.yticks_minor = axes.pois_yticks_minor

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
            x=lambda value: value * self.ensemble.spectral_radius,
            y=lambda value: value / np.pi / self.ensemble.spectral_radius,
        )

    @override
    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        self.build_figure()

        self.draw_histogram(
            color=self.histogram_color,
            alpha=self.histogram_alpha,
            zorder=self.histogram_zorder,
        )

        spectral_density = self.ensemble.spectral_density
        if (
            self.ensemble.max_spectral_polynomial_degree == 0
            and spectral_density.has_polynomial_expansion
        ):
            energies = np.linspace(*self.xlim, self.num_points)
            spectral_pdf = spectral_density.weight_pdf(energies)
            self.legend.labels = (
                self.legend.labels[0],
                _POLYNOMIAL_WEIGHT_LEGEND,
            )
        else:
            coefficients = self.calibration_coefficients("spectral")
            if coefficients is None:
                self.legend.handles = self.legend.handles[:1]
                self.legend.labels = self.legend.labels[:1]
                spectral_pdf = None
            else:
                energies = np.linspace(*self.xlim, self.num_points)
                spectral_pdf = spectral_density.variate_pdf(
                    energies,
                    coeffs=coefficients,
                )

        if spectral_pdf is not None:
            plot = cast(Callable[..., object], self.ax.plot)
            _ = plot(
                energies,
                spectral_pdf,
                color=self.pdf_color,
                alpha=self.pdf_alpha,
                linewidth=self.pdf_width,
                zorder=self.pdf_zorder,
            )

        self.finish_plot(path=path)


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class UnfoldedSpectralHistogramAxes(PlotAxes):
    xticks: tuple[float, ...] = (-0.5, 0.0, 0.5)  # units of dimension
    xticks_minor: tuple[float, ...] = (-0.25, 0.25)
    xlabel: str = r"$\xi / D$"
    xtick_labels: tuple[str, ...] = (
        r"$-0.5$",
        r"$0.0$",
        r"$+0.5$",
    )

    yticks: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5)  # units of dimension^{-1}
    yticks_minor: tuple[float, ...] = (0.25, 0.75, 1.25, 1.75)
    ylabel: str = r"$\ensavg{\rho(\xi)} D$"
    ytick_labels: tuple[str, ...] = (
        r"$0.0$",
        r"$0.5$",
        r"$1.0$",
        r"$1.5$",
    )


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class UnfoldedSpectralHistogramPlot(Plot):
    data: Data
    axes: PlotAxes = dataclasses.field(default_factory=UnfoldedSpectralHistogramAxes)
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
        self.ensemble: ManyBodyEnsemble = self.store_manifest_arg(
            "ensemble", ManyBodyEnsemble
        )
        dimension = self.ensemble.dimension

        self.legend: PlotLegend = PlotLegend(
            handles=self.legend_handles,
            labels=self.legend_labels,
            loc="upper right",
            bbox=(0.94, 0.95),
        )

        if not self.legend.title:
            unfolding_type = self.data.metadata["unfolding"]
            if unfolding_type != "weight":
                unfolding_degree = self.data.metadata["polynomial_degree"]
                self.legend.title = (
                    self.ensemble.to_latex
                    + f"\n{unfolding_type} unfolded, degree {unfolding_degree}"
                )
            else:
                self.legend.title = self.ensemble.to_latex + "\nweight unfolded"

        self.scale_limits_and_ticks(
            x=lambda value: value * dimension,
            y=lambda value: value / dimension,
        )

    @override
    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        self.build_figure()

        self.draw_histogram(
            color=self.histogram_color,
            alpha=self.histogram_alpha,
            zorder=self.histogram_zorder,
        )

        energies = np.linspace(self.xlim[0], self.xlim[1], self.num_points)

        dimension = self.ensemble.dimension
        unfolded_spectral_pdf = np.zeros(self.num_points)
        unfolded_spectral_pdf[np.abs(energies) < dimension / 2] = 1 / dimension

        plot = cast(Callable[..., object], self.ax.plot)
        _ = plot(
            energies,
            unfolded_spectral_pdf,
            color=self.pdf_color,
            alpha=self.pdf_alpha,
            linewidth=self.pdf_width,
            zorder=self.pdf_zorder,
        )

        self.finish_plot(path=path)
