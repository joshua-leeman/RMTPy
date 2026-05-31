from __future__ import annotations

import attrs
import numpy as np

from ..histogram import Histogram
from ..observable import Observable
from .observables import create_width_histograms


def compute_width_value(
    partial_widths: np.ndarray,
    width_index: tuple[int, ...],
) -> float:
    if len(width_index) == 2:
        return float(partial_widths[width_index[0]][width_index[1]])
    if len(width_index) == 1:
        return float(np.sum(partial_widths[width_index[0]]))
    raise ValueError("Invalid width index in histogram metadata.")


def create_partial_width_outputs(simulation) -> PartialWidthOutputs:
    return PartialWidthOutputs(histograms=tuple(create_width_histograms(simulation)))


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class PartialWidthOutputs:
    histograms: tuple[Observable[Histogram], ...]

    def add(self, partial_widths: np.ndarray) -> None:
        for observable in self.histograms:
            histogram = observable.data
            width_index = histogram.metadata["index"]
            width_value = compute_width_value(partial_widths, width_index)
            histogram.add_histogram_contribution(width_value)
            histogram.metadata["average_width"] += width_value

    def normalize_by_average_width(self, realizs: int) -> None:
        for observable in self.histograms:
            histogram = observable.data
            histogram.metadata["average_width"] /= realizs
            histogram.bins[:] /= histogram.metadata["average_width"]
