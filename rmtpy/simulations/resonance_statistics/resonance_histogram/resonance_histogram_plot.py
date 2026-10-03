import dataclasses
import math
from collections.abc import Callable
from pathlib import Path
from typing import cast, override

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from ....compounds import CompoundEnsemble
from ....ensembles import PoissonEnsemble, SachdevYeKitaevEnsemble
from ...base_data import Data
from ...base_plot import (
    ENSEMBLE_AVERAGED_CURVE_WIDTH,
    UNFOLDING_LABELS_BY_TYPE,
    Plot,
    PlotAxes,
    PlotLegend,
)
from .resonance_histogram_data import ResonanceHistogram

_Y_AXIS_PADDING: float = 0.05
_NUM_Y_MAJOR_INTERVALS: int = 5
_MAJOR_TICK_STEPS: tuple[float, ...] = (1.0, 2.0, 2.5, 5.0, 10.0)
_POLYNOMIAL_WEIGHT_LEGEND: str = "polynomial weight"


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

    _derived_attributes_are_set: bool = dataclasses.field(
        default=False,
        init=False,
        repr=False,
    )
    _resonance_centers: np.ndarray[tuple[int], np.dtype[np.floating]] | None = (
        dataclasses.field(
            default=None,
            init=False,
            repr=False,
        )
    )
    _resonance_pdf: np.ndarray[tuple[int], np.dtype[np.floating]] | None = (
        dataclasses.field(
            default=None,
            init=False,
            repr=False,
        )
    )

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
    pdf_width: float = ENSEMBLE_AVERAGED_CURVE_WIDTH
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
        if self._derived_attributes_are_set:
            return
        if not isinstance(self.data, ResonanceHistogram):
            raise ValueError("Data must be a `ResonanceHistogram` instance")

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

        self.axes.title = (
            "Resonance PDF: "
            + ensemble.to_latex
            + rf", $N_\text{{f}} = {{{self.compound.num_free_complex_fermions}}}$"
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
        )

        resonance_density = self.compound.resonance_density
        if (
            ensemble.max_spectral_polynomial_degree == 0
            and resonance_density.has_polynomial_expansion
        ):
            self._resonance_centers = np.linspace(*self.xlim, self.num_points)
            self._resonance_pdf = resonance_density.weight_pdf(self._resonance_centers)
            self.legend.labels = (
                self.legend.labels[0],
                _POLYNOMIAL_WEIGHT_LEGEND,
            )
        else:
            coefficients = self.calibration_coefficients("resonance")
            if coefficients is None:
                self.legend.handles = self.legend.handles[:1]
                self.legend.labels = self.legend.labels[:1]
            else:
                self._resonance_centers = np.linspace(*self.xlim, self.num_points)
                self._resonance_pdf = resonance_density.variate_pdf(
                    self._resonance_centers,
                    coeffs=coefficients,
                )

        if self._resonance_pdf is not None and not np.all(
            np.isfinite(self._resonance_pdf)
        ):
            raise ValueError("Resonance PDF values must be finite.")

        histogram_peak = float(np.max(self.data.histogram, initial=0.0))
        pdf_peak = (
            float(np.max(self._resonance_pdf, initial=0.0))
            if self._resonance_pdf is not None
            else 0.0
        )
        density_peak = max(histogram_peak, pdf_peak)
        if density_peak > 0.0:
            dimensionless_peak = np.pi * ensemble.spectral_radius * density_peak
            locator = MaxNLocator(
                nbins=_NUM_Y_MAJOR_INTERVALS,
                steps=_MAJOR_TICK_STEPS,
                min_n_ticks=3,
            )
            axes.yticks = tuple(
                float(value)
                for value in locator.tick_values(
                    0.0,
                    dimensionless_peak * (1.0 + _Y_AXIS_PADDING),
                )
            )
            self.ylim = (0.0, axes.yticks[-1])
            axes.yticks_minor = tuple(
                0.5 * (axes.yticks[index] + axes.yticks[index + 1])
                for index in range(len(axes.yticks) - 1)
            )

            major_tick_step = min(
                right - left
                for left, right in zip(axes.yticks, axes.yticks[1:], strict=False)
                if right > left
            )
            decimal_places = max(0, -math.floor(math.log10(major_tick_step)))
            scaled_step = major_tick_step * 10**decimal_places
            if not math.isclose(
                scaled_step,
                round(scaled_step),
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                decimal_places += 1

            zero_tolerance = major_tick_step * 1e-9
            axes.ytick_labels = tuple(
                rf"${(0.0 if abs(tick) <= zero_tolerance else tick):.{decimal_places}f}$"
                for tick in axes.yticks
            )

        self.scale_limits_and_ticks(
            y=lambda value: value / np.pi / ensemble.spectral_radius,
        )
        self._derived_attributes_are_set = True

    @override
    def plot(self, path: str | Path) -> None:
        if not self._derived_attributes_are_set:
            self.set_derived_attributes()

        self.build_figure()

        self.draw_histogram(
            color=self.histogram_color,
            alpha=self.histogram_alpha,
            zorder=self.histogram_zorder,
        )

        if self._resonance_centers is not None and self._resonance_pdf is not None:
            plot = cast(Callable[..., object], self.ax.plot)
            _ = plot(
                self._resonance_centers,
                self._resonance_pdf,
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
    ylabel: str = r"$D \ensavg{\rho(\xi)}$"
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
    pdf_width: float = ENSEMBLE_AVERAGED_CURVE_WIDTH
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

        unfolding_type = cast(str, self.data.metadata["unfolding"])
        unfolding_label = UNFOLDING_LABELS_BY_TYPE[unfolding_type]
        if unfolding_type != "weight":
            unfolding_degree = self.data.metadata["polynomial_degree"]
            title = f"{unfolding_label}({unfolding_degree})-unfolded"
        else:
            title = f"{unfolding_label}-unfolded"
        self.axes.title = (
            f"{title} Resonance PDF: {ensemble.to_latex}"
            + rf", $N_\text{{f}} = {{{self.compound.num_free_complex_fermions}}}$"
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
