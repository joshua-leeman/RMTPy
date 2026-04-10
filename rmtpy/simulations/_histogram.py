from __future__ import annotations

import numpy as np
from attrs import frozen, field

from ._data import Data


@frozen(kw_only=True, repr=False, eq=False, weakref_slot=False, getstate_setstate=False)
class Histogram(Data):
    support: tuple[float, float]
    scale: float = 1.0
    num_bins: int = 100

    bins: np.ndarray = field(init=False)

    @bins.default
    def _default_bins(self) -> np.ndarray:
        return self.scale * np.linspace(
            self.support[0], self.support[1], self.num_bins + 1
        )

    counts: np.ndarray = field(init=False)

    @counts.default
    def _default_counts(self) -> np.ndarray:
        return np.zeros(self.num_bins)

    histogram: np.ndarray = field(init=False)

    @histogram.default
    def _default_histogram(self) -> np.ndarray:
        return np.empty(self.num_bins)

    _realizs_count: int = field(init=False, factory=lambda: np.zeros((1,), dtype=int))

    @property
    def realizs(self) -> int:
        return self._realizs_count[0]

    def add_histogram_contribution(self, counts: np.ndarray) -> None:
        self.counts[:] += np.histogram(counts, bins=self.bins)[0]
        self._realizs_count[0] += 1

    def compute_histogram(self) -> None:
        self.histogram[:] = self.counts / np.sum(self.counts * np.diff(self.bins))
