import dataclasses
import math
from collections.abc import Callable
from pathlib import Path
from typing import cast, override

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullLocator

from .... import universal
from ....compounds import CompoundEnsemble
from ....conversion import json_value
from ....ensembles import ManyBodyEnsemble
from ...base_data import Data
from ...base_plot import (
    ConfigurableAxes,
    LogDimensionTimeAxes,
    LogDimensionUnfoldedTimeAxes,
    Plot,
    PlotAxes,
    PlotLegend,
)
from ...base_simulation import ExecutionState
from ...spectral_statistics import (
    SpectralStatisticsSimulation,
    load_spectral_statistics_simulation,
)
from ...spectral_statistics.spectral_form_factors import (
    FormFactorsData,
    FormFactorsPlot,
    UnfoldedFormFactorsPlot,
)
from ...statistics import LOG_D_TIME_SUPPORT, LOG_D_UNFOLDED_TIME_SUPPORT
from .time_delay_histogram_data import TimeDelayHistogram

type FormFactorsPlotType = FormFactorsPlot | UnfoldedFormFactorsPlot
type UnfoldingKey = tuple[str, int | None]


def format_energy_label(energy: float, energy_scale: float) -> str:
    scaled_energy = energy / energy_scale
    if np.isclose(scaled_energy, 0.0):
        return r"$E = 0$"

    return rf"$E = {scaled_energy:.2f}E_0$"


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class SpectralFormFactorsOverlay:
    simulation: SpectralStatisticsSimulation = dataclasses.field(
        repr=False,
    )

    form_factors: dict[UnfoldingKey, FormFactorsData] = dataclasses.field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.simulation.execution_state is not ExecutionState.COMPLETE:
            raise ValueError("Spectral-statistics simulation output is not complete.")

        form_factors: dict[UnfoldingKey, FormFactorsData] = {}
        self._store_form_factors(
            form_factors,
            key=("raw", None),
            data=self.simulation.raw_buffers.form_factors,
        )
        self._store_form_factors(
            form_factors,
            key=("weight", None),
            data=self.simulation.wgt_unfolded_buffers.form_factors,
        )

        for unfolding, buffers_collection in (
            ("averaged", self.simulation.ave_unfolded_buffers),
            ("variate", self.simulation.var_unfolded_buffers),
        ):
            for buffers in buffers_collection:
                polynomial_degree = buffers.polynomial_degree
                if polynomial_degree is None:
                    raise ValueError("Spectral form-factor unfolding degree is missing.")
                self._store_form_factors(
                    form_factors,
                    key=(unfolding, polynomial_degree),
                    data=buffers.form_factors,
                )

        self.form_factors = form_factors

    @classmethod
    def from_directory(
        cls,
        *,
        directory: str | Path,
        ensemble: ManyBodyEnsemble,
    ) -> SpectralFormFactorsOverlay:
        simulation = load_spectral_statistics_simulation(directory=directory)
        expected_configuration = cls._configuration_without_seed(ensemble)
        saved_configuration = cls._configuration_without_seed(simulation.ensemble)
        if saved_configuration != expected_configuration:
            raise ValueError(
                "Spectral-statistics ensemble configuration does not match the "
                + "time-delay ensemble configuration."
            )

        return cls(simulation=simulation)

    @staticmethod
    def _configuration_without_seed(
        ensemble: ManyBodyEnsemble,
    ) -> dict[str, object]:
        configuration = json_value(ensemble)
        if not isinstance(configuration, dict):
            raise TypeError("Ensemble configuration is malformed.")

        parameters = configuration.get("parameters")
        if not isinstance(parameters, dict):
            raise TypeError("Ensemble configuration parameters are malformed.")
        if "seed" not in parameters:
            raise ValueError("Ensemble configuration is missing `seed`.")

        comparable_parameters = dict(parameters)
        del comparable_parameters["seed"]
        return {**configuration, "parameters": comparable_parameters}

    @staticmethod
    def _unfolding_key(data: Data) -> UnfoldingKey:
        unfolding = data.metadata.get("unfolding")
        if unfolding not in {"raw", "weight", "averaged", "variate"}:
            raise ValueError("Plot-data unfolding metadata is malformed.")

        polynomial_degree = data.metadata.get("polynomial_degree")
        if unfolding in {"averaged", "variate"}:
            if (
                isinstance(polynomial_degree, bool)
                or not isinstance(polynomial_degree, int)
                or polynomial_degree <= 0
            ):
                raise ValueError("Plot-data polynomial degree is malformed.")
        elif polynomial_degree is not None:
            raise ValueError(
                "Raw and weight-unfolded plot data cannot have a polynomial degree."
            )

        return cast(str, unfolding), cast(int | None, polynomial_degree)

    @classmethod
    def _store_form_factors(
        cls,
        form_factors: dict[UnfoldingKey, FormFactorsData],
        *,
        key: UnfoldingKey,
        data: FormFactorsData,
    ) -> None:
        if cls._unfolding_key(data) != key:
            raise ValueError(
                "Spectral form-factor metadata does not match its simulation buffer."
            )
        if key in form_factors:
            raise ValueError(f"Spectral form-factor variant {key!r} is duplicated.")

        form_factors[key] = data

    def plot_for(self, data: TimeDelayHistogram) -> FormFactorsPlotType:
        form_factors = self.validate(data)
        key = self._unfolding_key(data)
        plot_cls = FormFactorsPlot if key == ("raw", None) else UnfoldedFormFactorsPlot
        return plot_cls(data=form_factors, context=self.simulation.manifest)

    def validate(self, data: TimeDelayHistogram) -> FormFactorsData:
        key = self._unfolding_key(data)
        try:
            form_factors = self.form_factors[key]
        except KeyError as exc:
            raise ValueError(f"Spectral form-factor variant {key!r} is missing.") from exc

        time_support = (data.bins[0], data.bins[-1])
        form_factor_support = (form_factors.times[0], form_factors.times[-1])

        if not np.allclose(time_support, form_factor_support, rtol=1e-12, atol=0.0):
            raise ValueError(
                "Spectral form-factor and time-delay data ranges do not match."
            )

        return form_factors


@dataclasses.dataclass(kw_only=True, eq=False, weakref_slot=False)
class _TimeDelayHistogramPlot(Plot):
    spectral_form_factors: SpectralFormFactorsOverlay | None = dataclasses.field(
        default=None,
        repr=False,
    )
    form_factors_plot: FormFactorsPlotType = dataclasses.field(
        init=False,
        repr=False,
    )
    form_factors_ax: Axes = dataclasses.field(
        init=False,
        repr=False,
    )

    def set_form_factors_derived_attributes(self) -> None:
        if self.spectral_form_factors is None:
            return
        if not isinstance(self.data, TimeDelayHistogram):
            raise ValueError("Data must be a `TimeDelayHistogram` instance")

        form_factors_plot = self.spectral_form_factors.plot_for(self.data)
        form_factors_plot.set_derived_attributes()
        if not np.allclose(self.xlim, form_factors_plot.xlim, rtol=1e-12, atol=0.0):
            raise ValueError(
                "Spectral form-factor and time-delay plot ranges do not match."
            )

        self.form_factors_plot = form_factors_plot
        self.legend.handles += form_factors_plot.legend.handles
        self.legend.labels += form_factors_plot.legend.labels
        axes = cast(TimeDelayHistogramAxes | UnfoldedTimeDelayHistogramAxes, self.axes)
        axes.right_ticks = False

    def draw_form_factors(self) -> None:
        if not hasattr(self, "form_factors_plot"):
            return

        form_factors_plot = self.form_factors_plot
        form_factors = cast(FormFactorsData, form_factors_plot.data)
        dimension = form_factors_plot.ensemble.dimension

        self.form_factors_ax = self.ax.twinx()

        set_yscale = cast(Callable[..., object], self.form_factors_ax.set_yscale)
        _ = set_yscale("log", base=dimension)

        set_y_major_locator = cast(
            Callable[..., object],
            self.form_factors_ax.yaxis.set_major_locator,
        )
        _ = set_y_major_locator(
            LogLocator(
                base=dimension,
                numticks=len(form_factors_plot.axes.yticks),
            )
        )

        plot = cast(Callable[..., object], self.form_factors_ax.plot)
        _ = plot(
            form_factors.times,
            form_factors.form_factor,
            color=form_factors_plot.sff_color,
            alpha=form_factors_plot.sff_alpha,
            linewidth=form_factors_plot.sff_width,
            zorder=form_factors_plot.sff_zorder,
        )
        _ = plot(
            form_factors.times,
            form_factors.connected_form_factor,
            color=form_factors_plot.csff_color,
            alpha=form_factors_plot.csff_alpha,
            linewidth=form_factors_plot.csff_width,
            zorder=form_factors_plot.csff_zorder,
        )

        if isinstance(form_factors_plot, UnfoldedFormFactorsPlot):
            universal_sff = form_factors_plot.ensemble.universal_connected_sff(
                form_factors.times
            )
            _ = plot(
                form_factors.times,
                universal_sff,
                color=form_factors_plot.universal_sff_color,
                alpha=form_factors_plot.universal_sff_alpha,
                linewidth=form_factors_plot.universal_sff_width,
                zorder=form_factors_plot.universal_sff_zorder,
            )

        if form_factors_plot.ylim:
            _ = self.form_factors_ax.set_ylim(form_factors_plot.ylim)

        self.configure_form_factors_axes()

    def configure_form_factors_axes(self) -> None:
        axes = self.form_factors_plot.axes
        for spine in self.form_factors_ax.spines.values():
            _ = spine.set_linewidth(axes.axes_width)

        if axes.ylabel:
            _ = self.form_factors_ax.set_ylabel(
                axes.ylabel,
                fontsize=axes.ylabel_fontsize,
            )
        if axes.yticks:
            _ = self.form_factors_ax.set_yticks(axes.yticks)

        if axes.yticks_minor:
            _ = self.form_factors_ax.set_yticks(axes.yticks_minor, minor=True)
        elif len(axes.yticks) > 1:
            _ = self.form_factors_ax.set_yticks(
                tuple(
                    math.sqrt(lower_tick * upper_tick)
                    for lower_tick, upper_tick in zip(
                        axes.yticks,
                        axes.yticks[1:],
                        strict=False,
                    )
                ),
                minor=True,
            )

        _ = self.form_factors_ax.tick_params(
            axis="y",
            direction="in",
            left=False,
            right=True,
            which="both",
            length=axes.tick_length,
            labelleft=False,
            labelright=True,
        )
        _ = self.form_factors_ax.tick_params(
            axis="y",
            which="minor",
            length=axes.tick_length / 2,
            labelleft=False,
            labelright=False,
        )

        if axes.ytick_labels:
            _ = self.form_factors_ax.set_yticklabels(
                axes.ytick_labels,
                fontsize=axes.tick_fontsize,
            )
        else:
            _ = self.form_factors_ax.tick_params(
                axis="y",
                labelsize=axes.tick_fontsize,
            )


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class TimeDelayHistogramAxes(LogDimensionTimeAxes):
    ylabel: str = r"$P(u)$"
    right_ticks: bool = True

    @override
    def configure(self, axes: ConfigurableAxes) -> None:
        super().configure(axes)
        if not self.right_ticks:
            _ = axes.tick_params(axis="y", right=False, labelright=False)


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class TimeDelayHistogramPlot(_TimeDelayHistogramPlot):
    data: Data

    axes: PlotAxes = dataclasses.field(
        default_factory=TimeDelayHistogramAxes,
    )

    num_points: int = 1000

    xlim: tuple[float, float] = LOG_D_TIME_SUPPORT

    histogram_zorder: int = 1
    histogram_alpha: float = 0.42
    histogram_color: str = "#7b2d26"

    pdf_zorder: int = 2
    pdf_width: float = 1.0
    pdf_alpha: float = 1.0
    pdf_color: str = "Black"
    pdf_legend: str = "BFB"

    def set_derived_attributes(self) -> None:
        self.compound: CompoundEnsemble = self.store_manifest_arg(
            "compound", CompoundEnsemble
        )
        mean_coupling_squared = cast(
            float, cast(object, np.mean(self.compound.couplings**2))
        )
        ensemble = self.compound.ensemble

        energy = cast(float, self.data.metadata["energy"])
        energy_label = format_energy_label(energy, ensemble.spectral_radius)
        self.legend_labels: tuple[str, str] = (energy_label, self.pdf_legend)
        self.legend_handles: tuple[Patch, Line2D] = (
            Patch(color=self.histogram_color, alpha=self.histogram_alpha),
            Line2D(
                [0],
                [0],
                color=self.pdf_color,
                alpha=self.pdf_alpha,
                linewidth=self.pdf_width,
            ),
        )

        self.legend: PlotLegend = PlotLegend(
            handles=self.legend_handles,
            labels=self.legend_labels,
            loc="upper right",
            bbox=(0.98, 0.95),
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

        scale = cast(float, self.data.metadata["scale"])
        self.scale_limits_and_ticks(
            x=lambda value: cast(float, ensemble.dimension**value * scale),
        )

    @override
    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        if not isinstance(self.data, TimeDelayHistogram):
            raise ValueError("Data must be a `TimeDelayHistogram` instance")
        self.set_form_factors_derived_attributes()

        self.build_figure()

        set_xscale = cast(Callable[..., object], self.ax.set_xscale)
        _ = set_xscale("log", base=self.compound.ensemble.dimension)

        set_major_locator = cast(Callable[..., object], self.ax.xaxis.set_major_locator)
        _ = set_major_locator(
            LogLocator(
                base=self.compound.ensemble.dimension,
                numticks=len(self.axes.xticks),
            )
        )
        set_minor_locator = cast(Callable[..., object], self.ax.xaxis.set_minor_locator)
        _ = set_minor_locator(NullLocator())

        self.draw_histogram(
            color=self.histogram_color,
            alpha=self.histogram_alpha,
            zorder=self.histogram_zorder,
        )

        times = np.geomspace(*self.xlim, self.num_points)
        time_delay_pdf = self.compound.time_delay_pdf(times)

        plot = cast(Callable[..., object], self.ax.plot)
        _ = plot(
            times,
            time_delay_pdf,
            color=self.pdf_color,
            alpha=self.pdf_alpha,
            linewidth=self.pdf_width,
            zorder=self.pdf_zorder,
        )

        self.draw_form_factors()
        self.finish_plot(path=path)


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class UnfoldedTimeDelayHistogramAxes(LogDimensionUnfoldedTimeAxes):
    ylabel: str = r"$P(\upsilon)$"
    right_ticks: bool = True

    @override
    def configure(self, axes: ConfigurableAxes) -> None:
        super().configure(axes)
        if not self.right_ticks:
            _ = axes.tick_params(axis="y", right=False, labelright=False)


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class UnfoldedTimeDelayHistogramPlot(_TimeDelayHistogramPlot):
    data: Data
    axes: PlotAxes = dataclasses.field(default_factory=UnfoldedTimeDelayHistogramAxes)
    num_points: int = 1000

    xlim: tuple[float, float] = LOG_D_UNFOLDED_TIME_SUPPORT

    histogram_zorder: int = 1
    histogram_alpha: float = 0.42
    histogram_color: str = "#7b2d26"

    pdf_zorder: int = 2
    pdf_width: float = 2.0
    pdf_alpha: float = 1.0
    pdf_color: str = "Black"
    pdf_legend: str = "BFB"

    def set_derived_attributes(self) -> None:
        self.compound: CompoundEnsemble = self.store_manifest_arg(
            "compound", CompoundEnsemble
        )
        mean_coupling_squared = float(np.mean(self.compound.couplings**2))
        ensemble = self.compound.ensemble

        energy = cast(float, self.data.metadata["energy"])
        energy_label = format_energy_label(energy, ensemble.spectral_radius)
        self.legend_labels: tuple[str, str] = (energy_label, self.pdf_legend)
        self.legend_handles: tuple[Patch, Line2D] = (
            Patch(color=self.histogram_color, alpha=self.histogram_alpha),
            Line2D(
                [0],
                [0],
                color=self.pdf_color,
                alpha=self.pdf_alpha,
                linewidth=self.pdf_width,
            ),
        )

        self.legend: PlotLegend = PlotLegend(
            handles=self.legend_handles,
            labels=self.legend_labels,
            loc="upper right",
            bbox=(0.98, 0.95),
        )

        coupling_exponent = cast(
            float, np.log10(mean_coupling_squared / ensemble.spectral_radius)
        )
        coupling_exponent = 0.0 if abs(coupling_exponent) < 0.005 else coupling_exponent
        coupling_label = rf"$\alpha = {{{coupling_exponent:.1f}}}$"
        if not self.legend.title:
            unfolding_type = self.data.metadata["unfolding"]
            if unfolding_type != "weight":
                polynomial_degree = self.data.metadata["polynomial_degree"]
                self.legend.title = (
                    ensemble.to_latex
                    + f"\n{unfolding_type} unfolded, degree {polynomial_degree}"
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

        scale = cast(float, self.data.metadata["scale"])
        self.scale_limits_and_ticks(
            x=lambda value: cast(float, ensemble.dimension**value * scale),
        )

    @override
    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        if not isinstance(self.data, TimeDelayHistogram):
            raise ValueError("Data must be a `TimeDelayHistogram` instance")
        self.set_form_factors_derived_attributes()

        self.build_figure()

        set_xscale = cast(Callable[..., object], self.ax.set_xscale)
        _ = set_xscale("log", base=self.compound.ensemble.dimension)

        set_major_locator = cast(Callable[..., object], self.ax.xaxis.set_major_locator)
        _ = set_major_locator(
            LogLocator(
                base=self.compound.ensemble.dimension,
                numticks=len(self.axes.xticks),
            )
        )
        set_minor_locator = cast(Callable[..., object], self.ax.xaxis.set_minor_locator)
        _ = set_minor_locator(NullLocator())

        self.draw_histogram(
            color=self.histogram_color,
            alpha=self.histogram_alpha,
            zorder=self.histogram_zorder,
        )

        times = np.geomspace(*self.xlim, self.num_points)
        time_delay_pdf = universal.time_delay_pdf(
            times,
            num_channels=self.compound.num_channels,
            heisenberg_time=2 * np.pi,
        )

        plot = cast(Callable[..., object], self.ax.plot)
        _ = plot(
            times,
            time_delay_pdf,
            color=self.pdf_color,
            alpha=self.pdf_alpha,
            linewidth=self.pdf_width,
            zorder=self.pdf_zorder,
        )

        self.draw_form_factors()
        self.finish_plot(path=path)
