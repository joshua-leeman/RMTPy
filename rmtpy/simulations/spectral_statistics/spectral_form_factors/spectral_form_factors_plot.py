import dataclasses
import math
from pathlib import Path
from typing import cast, override

import numpy as np
from matplotlib.lines import Line2D
from scipy.special import jn_zeros

from ....ensembles import ManyBodyEnsemble
from ...base_data import Data
from ...base_plot import (
    CONNECTED_FORM_FACTOR_COLOR,
    ENSEMBLE_AVERAGED_CURVE_WIDTH,
    FORM_FACTOR_COLOR,
    SINGLE_REALIZATION_CURVE_WIDTH,
    SINGLE_REALIZATION_FORM_FACTOR_COLOR,
    UNFOLDING_LABELS_BY_TYPE,
    ConfigurableAxes,
    LogDimensionTimeAxes,
    LogDimensionUnfoldedTimeAxes,
    Plot,
    PlotAxes,
    PlotLegend,
    configure_form_factor_axes,
    configure_form_factor_minor_ticks,
)
from ...statistics import LOG_D_TIME_SUPPORT, LOG_D_UNFOLDED_TIME_SUPPORT
from .spectral_form_factors_data import FormFactorsData


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class FormFactorsAxes(LogDimensionTimeAxes):
    yticks: tuple[float, ...] = (-2, -1, 0)  # log scale base dimension
    ytick_labels: tuple[str, ...] = (
        r"$D^{-2}$",
        r"$D^{-1}$",
        r"$1$",
    )

    @override
    def configure(self, axes: ConfigurableAxes) -> None:
        super().configure(axes)
        configure_form_factor_minor_ticks(axes)


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class FormFactorsPlot(Plot):
    data: Data
    axes: PlotAxes = dataclasses.field(default_factory=FormFactorsAxes)
    num_points: int = 1000

    xlim: tuple[float, float] = LOG_D_TIME_SUPPORT
    ylim: tuple[float, float] = (-2.2, 0.2)

    sff_zorder: int = 2
    sff_width: float = ENSEMBLE_AVERAGED_CURVE_WIDTH
    sff_alpha: float = 1.0
    sff_color: str = FORM_FACTOR_COLOR
    sff_legend: str = r"$K(u)$"

    csff_zorder: int = 2
    csff_width: float = ENSEMBLE_AVERAGED_CURVE_WIDTH
    csff_alpha: float = 1.0
    csff_color: str = CONNECTED_FORM_FACTOR_COLOR
    csff_legend: str = r"$K_{\text{\tiny conn}}(u)$"

    single_sff_zorder: int = 1
    single_sff_width: float = SINGLE_REALIZATION_CURVE_WIDTH
    single_sff_alpha: float = 1.0
    single_sff_color: str = SINGLE_REALIZATION_FORM_FACTOR_COLOR
    single_sff_legend: str = r"$K^{(1)}(u)$"

    legend_labels: tuple[str, str, str] = (
        sff_legend,
        csff_legend,
        single_sff_legend,
    )
    legend_handles: tuple[Line2D, Line2D, Line2D] = (
        Line2D([0], [0], color=sff_color, alpha=sff_alpha, linewidth=sff_width),
        Line2D([0], [0], color=csff_color, alpha=csff_alpha, linewidth=csff_width),
        Line2D(
            [0],
            [0],
            color=single_sff_color,
            alpha=single_sff_alpha,
            linewidth=single_sff_width,
        ),
    )

    def set_derived_attributes(self) -> None:
        if self._derived_attributes_are_set:
            return

        self.ensemble: ManyBodyEnsemble = self.store_manifest_arg(
            "ensemble", ManyBodyEnsemble
        )

        self.legend: PlotLegend = PlotLegend(
            handles=self.legend_handles,
            labels=self.legend_labels,
            loc="upper right",
            bbox=(0.925, 0.99),
        )
        self.axes.title = "Spectral Form Factors: " + self.ensemble.to_latex

        j_1_1 = cast(float, jn_zeros(1, 1)[0])
        self.scale_limits_and_ticks(
            x=lambda value: (
                math.pow(self.ensemble.dimension, value)
                * j_1_1
                / self.ensemble.spectral_radius
            ),
            y=lambda value: math.pow(self.ensemble.dimension, value),
        )

        self._derived_attributes_are_set: bool = True

    @override
    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        if not isinstance(self.data, FormFactorsData):
            raise ValueError("Data must be a `FormFactorsData` instance")

        self.build_figure()

        configure_form_factor_axes(
            self.ax,
            dimension=self.ensemble.dimension,
            x_tick_count=len(self.axes.xticks),
            y_tick_count=len(self.axes.yticks),
        )

        self.draw_curve(
            self.data.times,
            self.data.form_factor,
            color=self.sff_color,
            alpha=self.sff_alpha,
            width=self.sff_width,
            zorder=self.sff_zorder,
            label=self.sff_legend,
        )
        self.draw_curve(
            self.data.times,
            self.data.connected_form_factor,
            color=self.csff_color,
            alpha=self.csff_alpha,
            width=self.csff_width,
            zorder=self.csff_zorder,
            label=self.csff_legend,
        )
        if self.data.single_realization_form_factor_available:
            self.draw_curve(
                self.data.times,
                self.data.single_realization_form_factor,
                color=self.single_sff_color,
                alpha=self.single_sff_alpha,
                width=self.single_sff_width,
                zorder=self.single_sff_zorder,
                label=self.single_sff_legend,
            )

        else:
            self.legend.handles = self.legend.handles[:-1]
            self.legend.labels = self.legend.labels[:-1]

        self.finish_plot(path=path)


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class UnfoldedFormFactorsAxes(LogDimensionUnfoldedTimeAxes):
    yticks: tuple[float, ...] = (-2, -1, 0)  # log scale base dimension
    ytick_labels: tuple[str, ...] = (
        r"$D^{-2}$",
        r"$D^{-1}$",
        r"$1$",
    )

    @override
    def configure(self, axes: ConfigurableAxes) -> None:
        super().configure(axes)
        configure_form_factor_minor_ticks(axes)


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class UnfoldedFormFactorsPlot(Plot):
    data: Data
    axes: PlotAxes = dataclasses.field(default_factory=UnfoldedFormFactorsAxes)
    num_points: int = 1000

    xlim: tuple[float, float] = LOG_D_UNFOLDED_TIME_SUPPORT
    ylim: tuple[float, float] = (-2.2, 0.2)

    sff_zorder: int = 2
    sff_width: float = ENSEMBLE_AVERAGED_CURVE_WIDTH
    sff_alpha: float = 1.0
    sff_color: str = FORM_FACTOR_COLOR
    sff_legend: str = r"$K(\upsilon)$"

    csff_zorder: int = 2
    csff_width: float = ENSEMBLE_AVERAGED_CURVE_WIDTH
    csff_alpha: float = 1.0
    csff_color: str = CONNECTED_FORM_FACTOR_COLOR
    csff_legend: str = r"$K_{\text{\tiny conn}}(\upsilon)$"

    universal_sff_zorder: int = 2
    universal_sff_width: float = ENSEMBLE_AVERAGED_CURVE_WIDTH
    universal_sff_alpha: float = 1.0
    universal_sff_color: str = "Black"
    universal_sff_style: str = "dotted"
    universal_sff_legend: str = "universal"

    single_sff_zorder: int = 1
    single_sff_width: float = SINGLE_REALIZATION_CURVE_WIDTH
    single_sff_alpha: float = 1.0
    single_sff_color: str = SINGLE_REALIZATION_FORM_FACTOR_COLOR
    single_sff_legend: str = r"$K^{(1)}(\upsilon)$"

    legend_labels: tuple[str, str, str, str] = (
        sff_legend,
        csff_legend,
        universal_sff_legend,
        single_sff_legend,
    )
    legend_handles: tuple[Line2D, Line2D, Line2D, Line2D] = (
        Line2D([0], [0], color=sff_color, alpha=sff_alpha, linewidth=sff_width),
        Line2D([0], [0], color=csff_color, alpha=csff_alpha, linewidth=csff_width),
        Line2D(
            [0],
            [0],
            color=universal_sff_color,
            alpha=universal_sff_alpha,
            linewidth=universal_sff_width,
            linestyle=universal_sff_style,
        ),
        Line2D(
            [0],
            [0],
            color=single_sff_color,
            alpha=single_sff_alpha,
            linewidth=single_sff_width,
        ),
    )

    def set_derived_attributes(self) -> None:
        if self._derived_attributes_are_set:
            return

        self.ensemble: ManyBodyEnsemble = self.store_manifest_arg(
            "ensemble", ManyBodyEnsemble
        )

        if self.ensemble.universality_class is not None:
            self.universal_sff_legend = (
                rf"$K^{{\text{{\tiny {self.ensemble.universality_class}}}}}"
                + r"_{\text{\tiny conn}}(\upsilon)$"
            )
            self.legend_labels = (
                self.sff_legend,
                self.csff_legend,
                self.universal_sff_legend,
                self.single_sff_legend,
            )

        self.legend: PlotLegend = PlotLegend(
            handles=self.legend_handles,
            labels=self.legend_labels,
            loc="upper right",
            bbox=(0.925, 0.99),
        )

        unfolding_type = cast(str, self.data.metadata["unfolding"])
        unfolding_label = UNFOLDING_LABELS_BY_TYPE[unfolding_type]
        if unfolding_type != "weight":
            unfolding_degree = self.data.metadata["polynomial_degree"]
            title = f"{unfolding_label}({unfolding_degree})-unfolded"
        else:
            title = f"{unfolding_label}-unfolded"
        self.axes.title = f"{title} Spectral Form Factors: {self.ensemble.to_latex}"

        self.scale_limits_and_ticks(
            x=lambda value: math.pow(self.ensemble.dimension, value) * 2 * np.pi,
            y=lambda value: math.pow(self.ensemble.dimension, value),
        )

        self._derived_attributes_are_set: bool = True

    @override
    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        if not isinstance(self.data, FormFactorsData):
            raise ValueError("Data must be a `FormFactorsData` instance")

        self.build_figure()

        configure_form_factor_axes(
            self.ax,
            dimension=self.ensemble.dimension,
            x_tick_count=len(self.axes.xticks),
            y_tick_count=len(self.axes.yticks),
        )

        self.draw_curve(
            self.data.times,
            self.data.form_factor,
            color=self.sff_color,
            alpha=self.sff_alpha,
            width=self.sff_width,
            zorder=self.sff_zorder,
            label=self.sff_legend,
        )
        self.draw_curve(
            self.data.times,
            self.data.connected_form_factor,
            color=self.csff_color,
            alpha=self.csff_alpha,
            width=self.csff_width,
            zorder=self.csff_zorder,
            label=self.csff_legend,
        )

        universal_sff = self.ensemble.universal_connected_sff(self.data.times)
        self.draw_curve(
            self.data.times,
            universal_sff,
            color=self.universal_sff_color,
            alpha=self.universal_sff_alpha,
            width=self.universal_sff_width,
            style=self.universal_sff_style,
            zorder=self.universal_sff_zorder,
            label=self.universal_sff_legend,
        )
        if self.data.single_realization_form_factor_available:
            self.draw_curve(
                self.data.times,
                self.data.single_realization_form_factor,
                color=self.single_sff_color,
                alpha=self.single_sff_alpha,
                width=self.single_sff_width,
                zorder=self.single_sff_zorder,
                label=self.single_sff_legend,
            )

        else:
            self.legend.handles = self.legend.handles[:-1]
            self.legend.labels = self.legend.labels[:-1]

        self.finish_plot(path=path)
