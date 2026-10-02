import dataclasses
import math
from pathlib import Path
from typing import cast, override

import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from ....compounds import CompoundEnsemble
from ...base_data import Data
from ...base_plot import Plot, PlotAxes, PlotLegend
from .resonance_coefficients_histogram_data import ResonanceCoefficientsHistogram

_Y_AXIS_PADDING: float = 0.05
_NUM_MAJOR_INTERVALS: int = 5
_NUM_X_MAJOR_INTERVALS: int = 4
_MAJOR_TICK_STEPS: tuple[float, ...] = (1.0, 2.0, 2.5, 5.0, 10.0)
_CENTRAL_QUANTILES: tuple[float, float] = (0.01, 0.99)


def _nice_major_ticks(
    lower: float,
    upper: float,
    *,
    num_intervals: int = _NUM_MAJOR_INTERVALS,
    symmetric: bool = False,
) -> tuple[float, ...]:
    locator = MaxNLocator(
        nbins=num_intervals,
        steps=_MAJOR_TICK_STEPS,
        min_n_ticks=3,
        symmetric=symmetric,
    )
    return tuple(float(value) for value in locator.tick_values(lower, upper))


def _major_tick_step(major_ticks: tuple[float, ...]) -> float:
    return min(
        right - left
        for left, right in zip(major_ticks, major_ticks[1:], strict=False)
        if right > left
    )


def _minor_ticks(major_ticks: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(
        0.5 * (major_ticks[index] + major_ticks[index + 1])
        for index in range(len(major_ticks) - 1)
    )


def _latex_tick_labels(
    major_ticks: tuple[float, ...],
    *,
    show_positive_sign: bool,
) -> tuple[str, ...]:
    decimal_places = 0
    if len(major_ticks) > 1:
        step = _major_tick_step(major_ticks)
        decimal_places = max(0, -math.floor(math.log10(step)))
        scaled_step = step * 10**decimal_places
        if not math.isclose(
            scaled_step,
            round(scaled_step),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            decimal_places += 1

    step = major_ticks[1] - major_ticks[0] if len(major_ticks) > 1 else 1.0
    zero_tolerance = abs(step) * 1e-9

    labels: list[str] = []
    for tick in major_ticks:
        value = 0.0 if abs(tick) <= zero_tolerance else tick
        sign = "+" if show_positive_sign and value > 0.0 else ""
        labels.append(rf"${sign}{value:.{decimal_places}f}$")

    return tuple(labels)


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class ResonanceCoefficientsHistogramAxes(PlotAxes):
    pass


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class ResonanceCoefficientsHistogramPlot(Plot):
    data: Data
    axes: PlotAxes = dataclasses.field(
        default_factory=ResonanceCoefficientsHistogramAxes,
    )

    _derived_attributes_are_set: bool = dataclasses.field(
        default=False,
        init=False,
        repr=False,
    )

    histogram_zorder: int = 1
    histogram_alpha: float = 0.5
    histogram_color: str = "OrangeRed"
    histogram_legend: str = "simulation"

    legend_labels: tuple[str] = (histogram_legend,)
    legend_handles: tuple[Patch] = (Patch(color=histogram_color, alpha=histogram_alpha),)

    def set_derived_attributes(self) -> None:
        if not isinstance(self.data, ResonanceCoefficientsHistogram):
            raise ValueError("Data must be a `ResonanceCoefficientsHistogram` instance")

        coefficient_degree = cast(int, self.data.metadata["degree"])
        self.axes.xlabel = rf"$c_{{{coefficient_degree}}}$"
        self.axes.ylabel = rf"$P(c_{{{coefficient_degree}}})$"

        total_count = int(np.sum(self.data.counts))
        if total_count:
            cumulative_counts = np.cumsum(self.data.counts)
            lower_index, upper_index = np.searchsorted(
                cumulative_counts,
                np.multiply(_CENTRAL_QUANTILES, total_count),
                side="left",
            )
            lower = float(self.data.bins[lower_index])
            upper = float(self.data.bins[upper_index + 1])
        else:
            lower = float(self.data.bins[0])
            upper = float(self.data.bins[-1])

        occupied_extent = max(abs(lower), abs(upper))
        enclosing_ticks = _nice_major_ticks(
            -occupied_extent,
            occupied_extent,
            num_intervals=_NUM_X_MAJOR_INTERVALS,
            symmetric=True,
        )
        self.axes.xticks = enclosing_ticks
        horizontal_padding = 0.5 * _major_tick_step(self.axes.xticks)
        self.xlim = (
            self.axes.xticks[0] - horizontal_padding,
            self.axes.xticks[-1] + horizontal_padding,
        )
        self.axes.xticks_minor = _minor_ticks(self.axes.xticks)
        self.axes.xtick_labels = _latex_tick_labels(
            self.axes.xticks,
            show_positive_sign=True,
        )

        histogram_peak = float(np.max(self.data.histogram, initial=0.0))
        padded_peak = histogram_peak * (1.0 + _Y_AXIS_PADDING)
        y_tick_target = padded_peak if padded_peak > 0.0 else 1.0
        self.axes.yticks = _nice_major_ticks(
            0.0,
            y_tick_target,
        )
        self.ylim = (0.0, self.axes.yticks[-1])
        self.axes.yticks_minor = _minor_ticks(self.axes.yticks)
        self.axes.ytick_labels = _latex_tick_labels(
            self.axes.yticks,
            show_positive_sign=False,
        )

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
            self.legend.title = (
                ensemble.to_latex
                + "\n"
                + rf"$N_\text{{f}} = {{{self.compound.num_free_complex_fermions}}}$"
                + f", {{{coupling_label}}}"
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

        self.finish_plot(path=path)
