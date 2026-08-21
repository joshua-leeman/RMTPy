from __future__ import annotations

from typing import TYPE_CHECKING

import rmtpy.density

from ..histogram import Histogram, finalize_histogram
from ..observable import Observable
from ..statistics import create_observable
from .partial_width_histogram import PartialWidthHistogramPlot, TotalWidthHistogramPlot

if TYPE_CHECKING:
    from .partial_widths_statistics_simulation import PartialWidthsStatisticsSimulation


PARTIAL_WIDTH_LOG10_SUPPORT: rmtpy.density.Support = (-5.0, 2.0)

PARTIAL_WIDTH_NUM_BINS: int = 60

TOTAL_WIDTH_LOG10_SUPPORT: rmtpy.density.Support = (-2.0, 3.0)

TOTAL_WIDTH_NUM_BINS: int = 100

WIDTH_LOG_BASE: float = 10.0


def create_width_histogram_observable(
    width_index: tuple[int, ...],
    *,
    unfolding: str = "raw",
) -> Observable:
    if len(width_index) == 2:
        histogram = Histogram(
            file_name=(
                f"partial_width_state_{width_index[0]}_channel_{width_index[1]}_histogram"
            ),
            log_base=WIDTH_LOG_BASE,
            support=PARTIAL_WIDTH_LOG10_SUPPORT,
            num_bins=PARTIAL_WIDTH_NUM_BINS,
        )
        plot_cls = PartialWidthHistogramPlot
    elif len(width_index) == 1:
        histogram = Histogram(
            file_name=f"total_width_state_{width_index[0]}_histogram",
            log_base=WIDTH_LOG_BASE,
            support=TOTAL_WIDTH_LOG10_SUPPORT,
            num_bins=TOTAL_WIDTH_NUM_BINS,
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
