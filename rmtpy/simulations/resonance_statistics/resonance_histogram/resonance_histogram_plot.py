import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import cast, override

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ....compounds import CompoundEnsemble
from ....ensembles import PoissonEnsemble, SachdevYeKitaevEnsemble
from ...base_data import Data
from ...base_plot import Plot, PlotAxes, PlotLegend
from .resonance_histogram_data import ResonanceHistogram


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class ResonanceHistogramAxes(PlotAxes):
    xticks: tuple[float, ...] = (-1.0, 0.0, 1.0)  # units of energy_0
    xticks_minor: tuple[float, ...] = (-0.5, 0.5)
    xlabel: str = r"$\mathcal{E} / E_0$"
    xtick_labels: tuple[str, ...] = (
        r"$-1.0$",
        r"$0.0$",
        r"$+1.0$",
    )

    yticks: tuple[float, ...] = (0.0, 1.0, 2.0)  # units of 1 / (pi * energy_0)
    yticks_minor: tuple[float, ...] = (0.0, 1.0, 2.0)
    ylabel: str = r"$\pi E_0 \ensavg{\rho(\mathcal{E})}$"
    ytick_labels: tuple[str, ...] = (
        r"$0.0$",
        r"$1.0$",
        r"$2.0$",
    )

    pois_yticks: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0)
    pois_yticks_minor: tuple[float, ...] = (0.5, 1.5, 2.5)
    pois_ytick_labels: tuple[str, ...] = (
        r"$0.0$",
        r"$1.0$",
        r"$2.0$",
        r"$3.0$",
    )

    syk2_yticks: tuple[float, ...] = tuple(range(6))
    syk2_yticks_minor: tuple[float, ...] = tuple(value + 0.5 for value in range(6))
    syk2_ytick_labels: tuple[str, ...] = (
        r"$0.0$",
        r"$1.0$",
        r"$2.0$",
        r"$3.0$",
        r"$4.0$",
        r"$5.0$",
    )

    syk4_yticks: tuple[float, ...] = tuple(range(3))
    syk4_yticks_minor: tuple[float, ...] = tuple(value + 0.5 for value in range(3))
    syk4_ytick_labels: tuple[str, ...] = (
        r"$0.0$",
        r"$1.0$",
        r"$2.0$",
    )


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class ResonanceHistogramPlot(Plot):
    data: Data
    axes: PlotAxes = dataclasses.field(default_factory=ResonanceHistogramAxes)

    xlim: tuple[float, float] = (-1.2, 1.2)  # units of energy_0
    ylim: tuple[float, float] = (0.0, 2.6)  # units of 1 / (pi * energy_0)

    pois_ylim: tuple[float, float] = (0.0, 1.25)
    syk2_ylim: tuple[float, float] = (0.0, 4.0)
    syk4_ylim: tuple[float, float] = (0.0, 2.5)

    histogram_zorder: int = 1
    histogram_alpha: float = 0.5
    histogram_color: str = "OrangeRed"
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
        self.compound: CompoundEnsemble = self.store_manifest_arg(
            "compound", CompoundEnsemble
        )
        mean_coupling_squared = float(np.mean(self.compound.couplings**2))
        ensemble = self.compound.ensemble

        self.legend: PlotLegend = PlotLegend(
            handles=self.legend_handles,
            labels=self.legend_labels,
            loc="upper right",
            bbox=(0.99, 0.95),
        )

        coupling_exponent = cast(
            float, np.log10(mean_coupling_squared / ensemble.spectral_radius)
        )
        coupling_exponent = 0.0 if abs(coupling_exponent) < 0.005 else coupling_exponent
        coupling_label = rf"$\alpha = {{{coupling_exponent:.1f}}}$"
        if not self.legend.title:
            self.legend.title = (
                ensemble.to_latex
                + "\n"
                + rf"$N_\text{{f}} = {{{self.compound.num_free_complex_fermions}}}$"
                + f", {{{coupling_label}}}"
            )

        axes = cast(ResonanceHistogramAxes, self.axes)
        if isinstance(ensemble, PoissonEnsemble):
            self.ylim = self.pois_ylim

            axes.ytick_labels = axes.pois_ytick_labels
            axes.yticks = axes.pois_yticks
            axes.yticks_minor = axes.pois_yticks_minor

        elif isinstance(ensemble, SachdevYeKitaevEnsemble):
            if ensemble.q == 2:
                self.ylim = self.syk2_ylim

                axes.ytick_labels = axes.syk2_ytick_labels
                axes.yticks = axes.syk2_yticks
                axes.yticks_minor = axes.syk2_yticks_minor

            elif ensemble.q == 4:
                self.ylim = self.syk4_ylim

                axes.ytick_labels = axes.syk4_ytick_labels
                axes.yticks = axes.syk4_yticks
                axes.yticks_minor = axes.syk4_yticks_minor

        self.scale_limits_and_ticks(
            x=lambda value: value * ensemble.spectral_radius,
            y=lambda value: value / np.pi / ensemble.spectral_radius,
        )

    @override
    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        if not isinstance(self.data, ResonanceHistogram):
            raise ValueError("Data must be a `ResonanceHistogram` instance")

        self.build_figure()

        self.draw_histogram(
            color=self.histogram_color,
            alpha=self.histogram_alpha,
            zorder=self.histogram_zorder,
        )

        coefficients = self.calibration_coefficients("resonance")
        if coefficients is None:
            self.legend.handles = self.legend.handles[:1]
            self.legend.labels = self.legend.labels[:1]
        else:
            resonance_centers = np.linspace(*self.xlim, self.num_points)
            resonance_pdf = self.compound.resonance_density.variate_pdf(
                resonance_centers,
                coeffs=coefficients,
            )

            plot = cast(Callable[..., object], self.ax.plot)
            _ = plot(
                resonance_centers,
                resonance_pdf,
                color=self.pdf_color,
                alpha=self.pdf_alpha,
                linewidth=self.pdf_width,
                zorder=self.pdf_zorder,
            )

        self.finish_plot(path=path)


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class UnfoldedResonanceHistogramAxes(PlotAxes):
    xticks: tuple[float, ...] = (-0.5, 0.0, 0.5)  # units of dimension
    xticks_minor: tuple[float, ...] = (-0.25, 0.25)
    xlabel: str = r"$\xi / D$"
    xtick_labels: tuple[str, ...] = (
        r"$-0.5$",
        r"$0.0$",
        r"$+0.5$",
    )

    yticks: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5)
    yticks_minor: tuple[float, ...] = (0.25, 0.75, 1.25, 1.75)
    ylabel: str = r"$\ensavg{\rho(\xi)} D$"
    ytick_labels: tuple[str, ...] = (
        r"$0.0$",
        r"$0.5$",
        r"$1.0$",
        r"$1.5$",
    )


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class UnfoldedResonanceHistogramPlot(Plot):
    data: Data
    axes: PlotAxes = dataclasses.field(default_factory=UnfoldedResonanceHistogramAxes)
    num_points: int = 1000

    xlim: tuple[float, float] = (-0.6, 0.6)  # units of dimension
    ylim: tuple[float, float] = (0.0, 1.625)  # units of dimension^{-1}

    histogram_zorder: int = 1
    histogram_alpha: float = 0.5
    histogram_color: str = "OrangeRed"
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
        self.compound: CompoundEnsemble = self.store_manifest_arg(
            "compound", CompoundEnsemble
        )
        mean_coupling_squared = float(np.mean(self.compound.couplings**2))
        ensemble = self.compound.ensemble

        self.legend: PlotLegend = PlotLegend(
            handles=self.legend_handles,
            labels=self.legend_labels,
            loc="upper right",
            bbox=(0.94, 0.95),
        )

        coupling_exponent = cast(
            float, np.log10(mean_coupling_squared / ensemble.spectral_radius)
        )
        coupling_exponent = 0.0 if abs(coupling_exponent) < 0.005 else coupling_exponent
        coupling_label = rf"$\alpha = {{{coupling_exponent:.1f}}}$"
        if not self.legend.title:
            unfolding_type = self.data.metadata["unfolding"]
            if unfolding_type != "weight":
                unfolding_degree = self.data.metadata["polynomial_degree"]
                self.legend.title = (
                    ensemble.to_latex
                    + f"\n{unfolding_type} unfolded, degree {unfolding_degree}"
                    + "\n"
                    + rf"$N_\text{{f}} = {{{self.compound.num_free_complex_fermions}}}$"
                    + f", {{{coupling_label}}}"
                )
            else:
                self.legend.title = (
                    ensemble.to_latex
                    + "\nweight unfolded"
                    + "\n"
                    + rf"$N_\text{{f}} = {{{self.compound.num_free_complex_fermions}}}$"
                    + f", {{{coupling_label}}}"
                )

        self.scale_limits_and_ticks(
            x=lambda value: value * ensemble.dimension,
            y=lambda value: value / ensemble.dimension,
        )

    @override
    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        if not isinstance(self.data, ResonanceHistogram):
            raise ValueError("Data must be a `ResonanceHistogram` instance")

        self.build_figure()

        self.draw_histogram(
            color=self.histogram_color,
            alpha=self.histogram_alpha,
            zorder=self.histogram_zorder,
        )

        resonance_centers = np.linspace(*self.xlim, self.num_points)

        dimension = self.compound.ensemble.dimension
        unfolded_resonance_pdf = np.zeros(self.num_points)
        unfolded_resonance_pdf[np.abs(resonance_centers) < dimension / 2] = 1 / dimension

        plot = cast(Callable[..., object], self.ax.plot)
        _ = plot(
            resonance_centers,
            unfolded_resonance_pdf,
            color=self.pdf_color,
            alpha=self.pdf_alpha,
            linewidth=self.pdf_width,
            zorder=self.pdf_zorder,
        )

        self.finish_plot(path=path)
