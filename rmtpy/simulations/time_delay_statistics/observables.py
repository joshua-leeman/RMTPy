from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.special import jn_zeros

from ..histogram import Histogram
from ..observable import Observable
from ..statistics import create_histogram_observable
from .time_delay_histograms import (
    TimeDelayHistogramPlot,
    UnfoldedTimeDelayHistogramPlot,
)

if TYPE_CHECKING:
    from .time_delay_statistics_simulation import TimeDelayStatisticsSimulation

RAW_TIME_DELAY_LOGD_SUPPORT_DEFAULT: tuple[float, float] = (-0.5, 1.5)
UNFOLDED_TIME_DELAY_LOGD_SUPPORT_DEFAULT: tuple[float, float] = (-1.5, 0.5)
TIME_DELAY_NUM_BINS_DEFAULT: int = 100


def finalize_time_delay_histogram(histogram: Histogram) -> None:
    if np.sum(histogram.counts) == 0:
        histogram.histogram[:] = 0.0
        return

    histogram.normalize_histogram()


def compute_scaled_log_support(
    support: tuple[float, float],
    log_base: float,
    scale: float,
) -> tuple[float, float]:
    return tuple(endpoint + np.log(scale) / np.log(log_base) for endpoint in support)


def create_raw_time_delay_histogram_support(
    simulation: TimeDelayStatisticsSimulation,
) -> tuple[float, float]:
    energy_0: float = simulation.compound.ensemble.spectral_radius
    dimension: int = simulation.compound.ensemble.dimension
    scale: float = float(jn_zeros(1, 1)[0]) / energy_0
    return compute_scaled_log_support(
        RAW_TIME_DELAY_LOGD_SUPPORT_DEFAULT,
        dimension,
        scale,
    )


def create_unfolded_time_delay_histogram_support(
    simulation: TimeDelayStatisticsSimulation,
) -> tuple[float, float]:
    dimension: int = simulation.compound.ensemble.dimension
    return compute_scaled_log_support(
        UNFOLDED_TIME_DELAY_LOGD_SUPPORT_DEFAULT,
        dimension,
        2 * np.pi,
    )


def raw_time_delay_scale(simulation: TimeDelayStatisticsSimulation) -> float:
    return float(jn_zeros(1, 1)[0]) / simulation.compound.ensemble.spectral_radius


def unfolded_time_delay_scale(_: TimeDelayStatisticsSimulation) -> float:
    return 2 * np.pi


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
    support: tuple[float, float],
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
        num_bins=TIME_DELAY_NUM_BINS_DEFAULT,
        plot_cls=plot_cls,
        metadata=metadata,
        finalize=finalize_time_delay_histogram,
    )


def create_time_delay_histograms(
    simulation: TimeDelayStatisticsSimulation,
) -> list[Observable]:
    support: tuple[float, float] = create_raw_time_delay_histogram_support(simulation)
    scale: float = raw_time_delay_scale(simulation)
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


def create_unfolded_time_delay_histograms(
    *,
    simulation: TimeDelayStatisticsSimulation,
    file_name_prefix: str,
    unfolding: str,
    degree: int | None = None,
) -> list[Observable]:
    support: tuple[float, float] = create_unfolded_time_delay_histogram_support(
        simulation
    )
    scale: float = unfolded_time_delay_scale(simulation)
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


def create_weight_unfolded_time_delay_histograms(
    simulation: TimeDelayStatisticsSimulation,
) -> list[Observable]:
    return create_unfolded_time_delay_histograms(
        simulation=simulation,
        file_name_prefix="time_delay_histogram_weight_unfolded",
        unfolding="weight",
    )


def create_avg_unfolded_time_delay_histograms(
    simulation: TimeDelayStatisticsSimulation,
) -> list[Observable]:
    histograms: list[Observable] = []
    for degree in simulation.truncated_degrees:
        histograms.extend(
            create_unfolded_time_delay_histograms(
                simulation=simulation,
                file_name_prefix="time_delay_histogram_avg_unfolded",
                unfolding="avg",
                degree=degree,
            )
        )
    return histograms


def create_var_unfolded_time_delay_histograms(
    simulation: TimeDelayStatisticsSimulation,
) -> list[Observable]:
    histograms: list[Observable] = []
    for degree in simulation.truncated_degrees:
        histograms.extend(
            create_unfolded_time_delay_histograms(
                simulation=simulation,
                file_name_prefix="time_delay_histogram_var_unfolded",
                unfolding="var",
                degree=degree,
            )
        )
    return histograms
