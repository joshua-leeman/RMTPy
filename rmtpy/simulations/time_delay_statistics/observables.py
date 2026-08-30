from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.special import jn_zeros

import rmtpy.density

from ..histogram import Histogram
from ..observable import Observable
from ..plot import DIMENSION_TIME_LOG_SUPPORT, UNFOLDED_DIMENSION_TIME_LOG_SUPPORT
from ..statistics import create_histogram_observable
from .time_delay_histograms import (
    TimeDelayHistogramPlot,
    UnfoldedTimeDelayHistogramPlot,
)

if TYPE_CHECKING:
    from .time_delay_statistics_simulation import TimeDelayStatisticsSimulation

NUM_BINS: int = 100

RAW_LOG_D_TIME_DELAY_SUPPORT: rmtpy.density.Support = DIMENSION_TIME_LOG_SUPPORT

UNFOLDED_LOG_D_TIME_DELAY_SUPPORT: rmtpy.density.Support = (
    UNFOLDED_DIMENSION_TIME_LOG_SUPPORT
)


def compute_scaled_log_support(
    support: rmtpy.density.Support,
    *,
    log_base: float = 10.0,
    scale: float = 1.0,
) -> rmtpy.density.Support:
    return tuple(endpoint + np.log(scale) / np.log(log_base) for endpoint in support)


def create_raw_time_delay_histogram_support(
    simulation: TimeDelayStatisticsSimulation,
) -> rmtpy.density.Support:
    scale = float(jn_zeros(1, 1)[0]) / simulation.compound.ensemble.spectral_radius
    return compute_scaled_log_support(
        RAW_LOG_D_TIME_DELAY_SUPPORT,
        log_base=simulation.compound.ensemble.dimension,
        scale=scale,
    )


def create_time_delay_histogram_file_name(
    prefix: str,
    degree: int | None = None,
) -> str:
    if degree is None:
        return prefix
    return f"{prefix}_degree_{degree}"


def create_time_delay_histogram_observable(
    *,
    file_name_prefix: str,
    simulation: TimeDelayStatisticsSimulation,
    energy_index: int,
    energy: float,
    support: rmtpy.density.Support,
    scale: float,
    plot_cls: type[TimeDelayHistogramPlot],
    unfolding: str,
    degree: int | None = None,
) -> Observable:
    metadata: dict[str, Any] = {
        "energy": float(energy),
        "energy_index": int(energy_index),
        "scale": scale,
        "unfolding": unfolding,
    }
    if degree is not None:
        metadata["degree"] = degree

    return create_histogram_observable(
        file_name=create_time_delay_histogram_file_name(
            file_name_prefix,
            degree,
        ),
        support=support,
        log_base=simulation.compound.ensemble.dimension,
        num_bins=NUM_BINS,
        plot_cls=plot_cls,
        metadata=metadata,
        finalize=finalize_time_delay_histogram,
    )


def create_time_delay_histograms(
    simulation: TimeDelayStatisticsSimulation,
) -> list[Observable]:
    support = create_raw_time_delay_histogram_support(simulation)
    scale = raw_time_delay_scale(simulation)
    return [
        create_time_delay_histogram_observable(
            file_name_prefix="time_delay_histogram",
            simulation=simulation,
            energy_index=energy_index,
            energy=energy,
            support=support,
            scale=scale,
            plot_cls=TimeDelayHistogramPlot,
            unfolding="raw",
        )
        for energy_index, energy in enumerate(simulation.energies)
    ]


def create_unfolded_time_delay_histogram_support(
    simulation: TimeDelayStatisticsSimulation,
) -> rmtpy.density.Support:
    dimension = simulation.compound.ensemble.dimension
    return compute_scaled_log_support(
        UNFOLDED_LOG_D_TIME_DELAY_SUPPORT,
        log_base=dimension,
        scale=2 * np.pi,
    )


def create_unfolded_time_delay_histograms(
    *,
    simulation: TimeDelayStatisticsSimulation,
    file_name_prefix: str,
    unfolding: str,
    degree: int | None = None,
) -> list[Observable]:
    support = create_unfolded_time_delay_histogram_support(simulation)
    scale = unfolded_time_delay_scale(simulation)
    return [
        create_time_delay_histogram_observable(
            file_name_prefix=file_name_prefix,
            simulation=simulation,
            energy_index=energy_index,
            energy=energy,
            support=support,
            scale=scale,
            plot_cls=UnfoldedTimeDelayHistogramPlot,
            unfolding=unfolding,
            degree=degree,
        )
        for energy_index, energy in enumerate(simulation.energies)
    ]


def finalize_time_delay_histogram(histogram: Histogram) -> None:
    if np.sum(histogram.counts) == 0:
        histogram.histogram[:] = 0.0
        return

    histogram.compute_histogram()


def raw_time_delay_scale(simulation: TimeDelayStatisticsSimulation) -> float:
    return float(jn_zeros(1, 1)[0]) / simulation.compound.ensemble.spectral_radius


def unfolded_time_delay_scale(_: TimeDelayStatisticsSimulation) -> float:
    return 2 * np.pi
