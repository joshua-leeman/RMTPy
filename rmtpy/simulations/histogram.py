from collections.abc import Callable
from typing import cast

import attrs
import numpy as np
from numpy.typing import NDArray

from ..density import array_of_floats, compute_histogram
from ..validators import is_valid_support
from .base_data import Data

NUM_BINS: int = 100


def _build_empty_histogram(hist: Histogram) -> NDArray[np.float64]:
    return np.zeros(hist.num_bins, dtype=np.float64)


def _build_histogram_bins(hist: Histogram) -> NDArray[np.float64]:
    return array_of_floats(
        support=hist.support,
        num_pts=hist.num_bins + 1,
        log_base=hist.log_base,
    )


def _build_zeroed_histogram_counts(hist: Histogram) -> NDArray[np.int64]:
    return np.zeros(hist.num_bins, dtype=np.int64)


def _validate_bins(hist: Histogram, _: object, values: NDArray[np.float64]) -> None:
    if (
        values.shape != (hist.num_bins + 1,)
        or not np.issubdtype(values.dtype, np.floating)
        or not np.all(np.isfinite(values))
        or np.any(np.diff(values) <= 0.0)
    ):
        raise ValueError("Histogram bins do not match the declared bin count.")


def _validate_counts(hist: Histogram, _: object, values: NDArray[np.int64]) -> None:
    if (
        values.shape != (hist.num_bins,)
        or not np.issubdtype(values.dtype, np.integer)
        or np.any(values < 0)
    ):
        raise ValueError("Histogram counts do not match the declared bin count.")


def _validate_histogram(hist: Histogram, _: object, values: NDArray[np.float64]) -> None:
    if (
        values.shape != (hist.num_bins,)
        or not np.issubdtype(values.dtype, np.floating)
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
    ):
        raise ValueError("Histogram values do not match the declared bin count.")


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class Histogram(Data):
    support: tuple[float, float] = attrs.field(
        converter=cast(Callable[[object], tuple[float, float]], tuple),
        validator=is_valid_support,
    )
    log_base: float | None = attrs.field(
        default=None,
        converter=attrs.converters.optional(float),
        validator=attrs.validators.optional(attrs.validators.gt(0.0)),
    )
    num_bins: int = attrs.field(
        default=NUM_BINS,
        validator=(
            attrs.validators.instance_of(int),
            attrs.validators.gt(0),
        ),
        repr=False,
    )

    bins: NDArray[np.float64] = attrs.field(
        default=attrs.Factory(_build_histogram_bins, takes_self=True),
        converter=np.asarray,
        validator=_validate_bins,
        repr=False,
    )
    counts: NDArray[np.int64] = attrs.field(
        default=attrs.Factory(_build_zeroed_histogram_counts, takes_self=True),
        converter=np.asarray,
        validator=_validate_counts,
        repr=False,
    )
    histogram: NDArray[np.float64] = attrs.field(
        default=attrs.Factory(_build_empty_histogram, takes_self=True),
        converter=np.asarray,
        validator=_validate_histogram,
        repr=False,
    )
    realizs: int = attrs.field(
        default=0,
        converter=int,
        validator=attrs.validators.ge(0),
    )

    def add_histogram_contribution(self, data: NDArray[np.float64], /) -> None:
        if isinstance(data, (int, float)):
            data = np.array([data], dtype=np.float64)

        indices = np.searchsorted(self.bins, data, side="right") - 1
        valid = (indices >= 0) & (indices < len(self.counts))
        np.add.at(self.counts, indices[valid], 1)
        object.__setattr__(self, "realizs", self.realizs + 1)

    def compute_histogram(self) -> None:
        if np.sum(self.counts) == 0:
            self.histogram.fill(0.0)
            return

        self.histogram[:] = compute_histogram(self.counts, bins=self.bins)

    def compute_histogram_as_probabilities(self) -> None:
        total = np.sum(self.counts)
        if total == 0:
            self.histogram.fill(0.0)
            return

        self.histogram[:] = self.counts / total
