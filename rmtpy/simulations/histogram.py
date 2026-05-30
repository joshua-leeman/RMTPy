from __future__ import annotations

import attrs
import numpy as np

import rmtpy.density
import rmtpy.validators

from .data import Data

NUM_BINS_DEFAULT: int = 100


def create_histogram_bins(hist: Histogram) -> np.ndarray:
    return rmtpy.density.array_of_floats(hist.support, hist.num_bins + 1, hist.log_base)


def create_zeroed_histogram_counts(hist: Histogram) -> np.ndarray:
    return np.zeros(hist.num_bins, dtype=np.int64)


def create_empty_histogram(hist: Histogram) -> np.ndarray:
    return np.empty(hist.num_bins, dtype=np.float64)


def finalize_histogram(hist: Histogram) -> None:
    hist.normalize_histogram()


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class Histogram(Data):
    support: tuple[float, float] = attrs.field(
        converter=tuple,
        validator=lambda _, __, support: rmtpy.validators.validate_support(support),
    )
    log_base: float | None = attrs.field(
        default=None,
        converter=attrs.converters.optional(float),
        validator=attrs.validators.optional(attrs.validators.gt(0.0)),
    )
    num_bins: int = attrs.field(
        default=NUM_BINS_DEFAULT,
        validator=[
            attrs.validators.instance_of(int),
            attrs.validators.gt(0),
        ],
        repr=False,
    )

    bins: np.ndarray = attrs.field(
        default=attrs.Factory(create_histogram_bins, takes_self=True),
        init=False,
        repr=False,
    )
    counts: np.ndarray = attrs.field(
        default=attrs.Factory(create_zeroed_histogram_counts, takes_self=True),
        init=False,
        repr=False,
    )
    histogram: np.ndarray = attrs.field(
        default=attrs.Factory(create_empty_histogram, takes_self=True),
        init=False,
        repr=False,
    )

    _realizs_count: int = attrs.field(
        init=False, factory=lambda: np.zeros((1,), dtype=np.int64)
    )

    @property
    def realizs(self) -> int:
        return self._realizs_count[0]

    def add_histogram_contribution(self, data: np.ndarray) -> None:
        if isinstance(data, (int, float)):
            data = np.array([data], dtype=np.float64)

        indices = np.searchsorted(self.bins, data, side="right") - 1
        valid = (indices >= 0) & (indices < len(self.counts))
        np.add.at(self.counts, indices[valid], 1)
        self._realizs_count[0] += 1

    def normalize_histogram(self) -> None:
        self.histogram[:] = rmtpy.density.normalize_histogram(self.counts, self.bins)

    def compute_histogram_as_probabilities(self) -> None:
        self.histogram[:] = self.counts / np.sum(self.counts)
