import dataclasses
from abc import ABC
from collections.abc import Callable
from pathlib import Path
from typing import cast, override

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, LogLocator, NullLocator

from .... import universal
from ....compounds import CompoundEnsemble
from ....conversion import json_value
from ....ensembles import ManyBodyEnsemble
from ...base_data import Data
from ...base_plot import (
    ENSEMBLE_AVERAGED_CURVE_WIDTH,
    UNFOLDING_LABELS_BY_TYPE,
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

TIME_DELAY_HISTOGRAM_COLOR: str = "#E8B03F"


def format_energy_label(energy: float, energy_scale: float) -> str:
    scaled_energy = energy / energy_scale
    if np.isclose(scaled_energy, 0.0):
        return r"$E = 0$"

    return rf"$E = {scaled_energy:.2f}E_0$"


def _configuration_without_seed(
    ensemble: ManyBodyEnsemble,
) -> dict[str, object]:
    configuration = json_value(ensemble)
    if not isinstance(configuration, dict):
        raise TypeError("Ensemble configuration is malformed.")

    configuration = cast(dict[str, object], configuration)
    parameters = configuration.get("parameters")
    if not isinstance(parameters, dict):
        raise TypeError("Ensemble configuration parameters are malformed.")
    if "seed" not in parameters:
        raise ValueError("Ensemble configuration is missing `seed`.")

    comparable_parameters = dict(cast(dict[str, object], parameters))
    del comparable_parameters["seed"]
    return {**configuration, "parameters": comparable_parameters}


def _unfolding_key(data: Data) -> UnfoldingKey:
    unfolding = data.metadata.get("unfolding")
    if unfolding not in {"raw", "weight", "average", "variate"}:
        raise ValueError("Plot-data unfolding metadata is malformed.")

    polynomial_degree = data.metadata.get("polynomial_degree")
    if unfolding in {"average", "variate"}:
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

    return cast(str, unfolding), polynomial_degree


def _store_form_factors(
    form_factors: dict[UnfoldingKey, FormFactorsData],
    *,
    key: UnfoldingKey,
    data: FormFactorsData,
) -> None:
    if _unfolding_key(data) != key:
        raise ValueError(
            "Spectral form-factor metadata does not match its simulation buffer."
        )
    if key in form_factors:
        raise ValueError(f"Spectral form-factor variant {key!r} is duplicated.")

    form_factors[key] = data


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
        _store_form_factors(
            form_factors,
            key=("raw", None),
            data=self.simulation.raw_buffers.form_factors,
        )
        _store_form_factors(
            form_factors,
            key=("weight", None),
            data=self.simulation.wgt_unfolded_buffers.form_factors,
        )

        for unfolding, buffers_collection in (
            ("average", self.simulation.ave_unfolded_buffers),
            ("variate", self.simulation.var_unfolded_buffers),
        ):
            for buffers in buffers_collection:
                polynomial_degree = buffers.polynomial_degree
                if polynomial_degree is None:
                    raise ValueError("Spectral form-factor unfolding degree is missing.")
                _store_form_factors(
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
        expected_configuration = _configuration_without_seed(ensemble)
        saved_configuration = _configuration_without_seed(simulation.ensemble)
        if saved_configuration != expected_configuration:
            raise ValueError(
                "Spectral-statistics ensemble configuration does not match the "
                + "time-delay ensemble configuration."
            )

        return cls(simulation=simulation)

    def plot_for(self, data: TimeDelayHistogram) -> FormFactorsPlotType:
        form_factors = self.validate(data)
        key = _unfolding_key(data)
        plot_cls = FormFactorsPlot if key == ("raw", None) else UnfoldedFormFactorsPlot
        return plot_cls(data=form_factors, context=self.simulation.manifest)

    def validate(self, data: TimeDelayHistogram) -> FormFactorsData:
        key = _unfolding_key(data)
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
class _TimeDelayHistogramPlot(Plot, ABC):
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
        self.legend.handles += form_factors_plot.legend.handles[:-1]
        self.legend.labels += form_factors_plot.legend.labels[:-1]
        axes = cast(TimeDelayHistogramAxes | UnfoldedTimeDelayHistogramAxes, self.axes)
        axes.right_ticks = False

    def draw_form_factors(self) -> None:
        if not hasattr(self, "form_factors_plot"):
            return

        form_factors_plot = self.form_factors_plot
        form_factors = cast(FormFactorsData, form_factors_plot.data)
        dimension = form_factors_plot.ensemble.dimension

        self.form_factors_ax = cast(Callable[[], Axes], self.ax.twinx)()

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
                linestyle=form_factors_plot.universal_sff_style,
                zorder=form_factors_plot.universal_sff_zorder,
            )

        if form_factors_plot.ylim:
            _ = self.form_factors_ax.set_ylim(form_factors_plot.ylim)

        self.configure_form_factors_axes()

    def configure_form_factors_axes(self) -> None:
        axes = self.form_factors_plot.axes
        form_factors_axes = cast(ConfigurableAxes, cast(object, self.form_factors_ax))
        for spine in form_factors_axes.spines.values():
            _ = spine.set_linewidth(axes.axes_width)

        _ = form_factors_axes.set_ylabel(
            "SFFs",
            fontsize=axes.ylabel_fontsize,
            rotation=270,
            labelpad=15,
        )
        if axes.yticks:
            _ = form_factors_axes.set_yticks(axes.yticks)

        _ = form_factors_axes.tick_params(
            axis="y",
            direction="in",
            left=False,
            right=True,
            which="major",
            length=axes.tick_length,
            labelleft=False,
            labelright=True,
        )
        _ = form_factors_axes.tick_params(
            axis="y",
            which="minor",
            left=False,
            right=False,
            labelleft=False,
            labelright=False,
        )

        if axes.ytick_labels:
            _ = form_factors_axes.set_yticklabels(
                axes.ytick_labels,
                fontsize=axes.tick_fontsize,
            )
        else:
            _ = form_factors_axes.tick_params(
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
    histogram_color: str = TIME_DELAY_HISTOGRAM_COLOR

    pdf_zorder: int = 2
    pdf_width: float = ENSEMBLE_AVERAGED_CURVE_WIDTH
    pdf_alpha: float = 1.0
    pdf_color: str = "Black"
    pdf_legend: str = "BFB"

    def set_derived_attributes(self) -> None:
        if self._derived_attributes_are_set:
            return

        self.compound: CompoundEnsemble = self.store_manifest_arg(
            "compound", CompoundEnsemble
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
            bbox=(0.98, 0.99),
        )

        if self.spectral_form_factors is None:
            descriptive_title = "Time-delay PDF"
        else:
            descriptive_title = "Time-delay PDF vs SFFs"

        self.axes.title = (
            f"{descriptive_title}: "
            + ensemble.to_latex
            + rf", $N_\text{{f}} = {{{self.compound.num_free_complex_fermions}}}$"
        )

        scale = cast(float, self.data.metadata["scale"])
        self.scale_limits_and_ticks(
            x=lambda value: cast(float, ensemble.dimension**value * scale),
        )

        self._derived_attributes_are_set: bool = True

    @override
    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        if not isinstance(self.data, TimeDelayHistogram):
            raise ValueError("Data must be a `TimeDelayHistogram` instance")
        self.set_form_factors_derived_attributes()

        self.build_figure()

        self.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

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
    histogram_color: str = TIME_DELAY_HISTOGRAM_COLOR

    pdf_zorder: int = 2
    pdf_width: float = ENSEMBLE_AVERAGED_CURVE_WIDTH
    pdf_alpha: float = 1.0
    pdf_color: str = "Black"
    pdf_legend: str = "BFB"

    def set_derived_attributes(self) -> None:
        if self._derived_attributes_are_set:
            return

        self.compound: CompoundEnsemble = self.store_manifest_arg(
            "compound", CompoundEnsemble
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
            bbox=(0.98, 0.99),
        )

        if self.spectral_form_factors is None:
            descriptive_title = "Time-delay PDF"
        else:
            descriptive_title = "Time-delay PDF vs SFFs"

        unfolding_type = cast(str, self.data.metadata["unfolding"])
        unfolding_label = UNFOLDING_LABELS_BY_TYPE[unfolding_type]
        if unfolding_type != "weight":
            unfolding_degree = self.data.metadata["polynomial_degree"]
            title = f"{unfolding_label}({unfolding_degree})-unfolded"
        else:
            title = f"{unfolding_label}-unfolded"
        self.axes.title = (
            f"{title} {descriptive_title}: {ensemble.to_latex}"
            + rf", $N_\text{{f}} = {{{self.compound.num_free_complex_fermions}}}$"
        )

        scale = cast(float, self.data.metadata["scale"])
        self.scale_limits_and_ticks(
            x=lambda value: cast(float, ensemble.dimension**value * scale),
        )

        self._derived_attributes_are_set: bool = True

    @override
    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        if not isinstance(self.data, TimeDelayHistogram):
            raise ValueError("Data must be a `TimeDelayHistogram` instance")
        self.set_form_factors_derived_attributes()

        self.build_figure()

        self.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

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
