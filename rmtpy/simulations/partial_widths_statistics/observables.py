from __future__ import annotations

from typing import TYPE_CHECKING

from ..histogram import Histogram, finalize_histogram
from ..observable import Observable
from ..statistics import create_observable
from .partial_width_histogram import PartialWidthHistogramPlot, TotalWidthHistogramPlot

if TYPE_CHECKING:
    from .partial_widths_statistics_simulation import PartialWidthsStatisticsSimulation

WIDTH_LOG_BASE_DEFAULT: float = 10.0
PARTIAL_WIDTH_LOG10_SUPPORT_DEFAULT: tuple[float, float] = (-5.0, 2.0)
TOTAL_WIDTH_LOG10_SUPPORT_DEFAULT: tuple[float, float] = (-2.0, 3.0)
PARTIAL_WIDTH_NUM_BINS_DEFAULT: int = 60
TOTAL_WIDTH_NUM_BINS_DEFAULT: int = 100


def create_width_histogram_observable(
    width_index: tuple[int, ...],
    *,
    unfolding: str = "raw",
) -> Observable:
    if len(width_index) == 2:
        histogram: Histogram = Histogram(
            file_name=f"partial_width_{width_index[0]}{width_index[1]}_histogram",
            log_base=WIDTH_LOG_BASE_DEFAULT,
            support=PARTIAL_WIDTH_LOG10_SUPPORT_DEFAULT,
            num_bins=PARTIAL_WIDTH_NUM_BINS_DEFAULT,
        )
        plot_cls: type = PartialWidthHistogramPlot
    elif len(width_index) == 1:
        histogram = Histogram(
            file_name=f"total_width_{width_index[0]}_histogram",
            log_base=WIDTH_LOG_BASE_DEFAULT,
            support=TOTAL_WIDTH_LOG10_SUPPORT_DEFAULT,
            num_bins=TOTAL_WIDTH_NUM_BINS_DEFAULT,
        )
        plot_cls = TotalWidthHistogramPlot
    else:
        raise ValueError("Invalid width index in width_indices.")

    return create_observable(
        data=histogram,
        plot_cls=plot_cls,
        metadata={
            "index": width_index,
            "average_width": 0.0,
            "unfolding": unfolding,
        },
        finalize=finalize_histogram,
    )


def create_width_histograms(
    simulation: PartialWidthsStatisticsSimulation,
    *,
    unfolding: str = "raw",
) -> list[Observable]:
    return [
        create_width_histogram_observable(width_index, unfolding=unfolding)
        for width_index in simulation.width_indices
    ]
