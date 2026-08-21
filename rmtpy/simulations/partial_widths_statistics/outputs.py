from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import attrs
import numpy as np

from ..histogram import Histogram
from ..observable import Observable
from .observables import create_width_histograms

if TYPE_CHECKING:
    from .partial_widths_statistics_simulation import PartialWidthsStatisticsSimulation


def compute_width_value(
    partial_widths: np.ndarray,
    width_index: tuple[int, ...],
) -> float:
    if len(width_index) == 2:
        return float(partial_widths[width_index[0]][width_index[1]])
    if len(width_index) == 1:
        return float(np.sum(partial_widths[width_index[0]]))
    raise ValueError("Invalid width index in histogram metadata.")


def create_partial_width_outputs(
    simulation: PartialWidthsStatisticsSimulation,
) -> PartialWidthOutputs:
    return PartialWidthOutputs(histograms=tuple(create_width_histograms(simulation)))


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class PartialWidthOutputs:
    """Width histograms in the same order as the requested index selections."""

    histograms: tuple[Observable[Histogram], ...]

    def iter_observables(self) -> Iterator[Observable]:
        yield from self.histograms

    def add(self, partial_widths: np.ndarray) -> None:
        for observable in self.histograms:
            histogram = observable.data
            width_index = histogram.metadata["index"]
            width_value = compute_width_value(partial_widths, width_index)
            histogram.add_histogram_contribution(width_value)
            histogram.metadata["average_width"] += width_value

    def normalize_by_average_width(self, realizs: int) -> None:
        average_widths: list[float] = []
        for observable in self.histograms:
            histogram = observable.data
            average_width = histogram.metadata["average_width"] / realizs
            if not np.isfinite(average_width) or average_width <= 0.0:
                width_index = histogram.metadata["index"]
                raise ValueError(
                    f"Average width for index {width_index} must be positive and finite."
                )
            average_widths.append(average_width)

        for observable, average_width in zip(
            self.histograms,
            average_widths,
            strict=True,
        ):
            histogram = observable.data
            histogram.metadata["average_width"] = average_width
            histogram.bins[:] /= average_width
