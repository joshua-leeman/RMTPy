from collections.abc import Callable
from functools import partial
from typing import cast

import attrs
import numpy as np
from numpy.typing import NDArray

from ..density import array_of_floats, compute_bin_centers
from ..validators import is_valid_support
from .base_data import Data

NUM_BINS: int = 100


def _build_empty_histogram2D(hist: Histogram2D) -> NDArray[np.float64]:
    return np.zeros((hist.x_num_bins, hist.y_num_bins), dtype=np.float64)


def _build_histogram2D_bins(hist: Histogram2D, axis: str) -> NDArray[np.float64]:
    if axis.strip().lower() == "x":
        return array_of_floats(
            support=hist.x_support,
            num_pts=hist.x_num_bins + 1,
            log_base=hist.x_log_base,
        )

    elif axis.strip().lower() == "y":
        return array_of_floats(
            support=hist.y_support,
            num_pts=hist.y_num_bins + 1,
            log_base=hist.y_log_base,
        )

    else:
        raise ValueError("`axis` must be either 'x' and 'y'.")


def _build_zeroed_histogram2D_counts(hist: Histogram2D) -> NDArray[np.int64]:
    return np.zeros((hist.x_num_bins, hist.y_num_bins), dtype=np.int64)


def _validate_x_bins(hist: Histogram2D, _: object, values: NDArray[np.float64]) -> None:
    if (
        values.shape != (hist.x_num_bins + 1,)
        or not np.issubdtype(values.dtype, np.floating)
        or not np.all(np.isfinite(values))
        or np.any(np.diff(values) <= 0.0)
    ):
        raise ValueError("2-D histogram x bins do not match the declared bin count.")


def _validate_y_bins(hist: Histogram2D, _: object, values: NDArray[np.float64]) -> None:
    if (
        values.shape != (hist.y_num_bins + 1,)
        or not np.issubdtype(values.dtype, np.floating)
        or not np.all(np.isfinite(values))
        or np.any(np.diff(values) <= 0.0)
    ):
        raise ValueError("2-D histogram y bins do not match the declared bin count.")


def _validate_counts(hist: Histogram2D, _: object, values: NDArray[np.int64]) -> None:
    if (
        values.shape != (hist.x_num_bins, hist.y_num_bins)
        or not np.issubdtype(values.dtype, np.integer)
        or np.any(values < 0)
    ):
        raise ValueError("2-D histogram counts do not match declared bin counts.")


def _validate_histogram(
    hist: Histogram2D, _: object, values: NDArray[np.float64]
) -> None:
    if (
        values.shape != (hist.x_num_bins, hist.y_num_bins)
        or not np.issubdtype(values.dtype, np.floating)
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
    ):
        raise ValueError("2-D histogram values do not match declared bin counts.")


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class Histogram2D(Data):
    x_support: tuple[float, float] = attrs.field(
        converter=cast(Callable[[object], tuple[float, float]], tuple),
        validator=is_valid_support,
    )
    x_log_base: float | None = attrs.field(
        default=None,
        converter=attrs.converters.optional(float),
        validator=attrs.validators.optional(attrs.validators.gt(0.0)),
    )
    x_num_bins: int = attrs.field(
        default=NUM_BINS,
        validator=(
            attrs.validators.instance_of(int),
            attrs.validators.gt(0),
        ),
        repr=False,
    )

    y_support: tuple[float, float] = attrs.field(
        converter=cast(Callable[[object], tuple[float, float]], tuple),
        validator=is_valid_support,
    )
    y_log_base: float | None = attrs.field(
        default=None,
        converter=attrs.converters.optional(float),
        validator=attrs.validators.optional(attrs.validators.gt(0.0)),
    )
    y_num_bins: int = attrs.field(
        default=NUM_BINS,
        validator=(
            attrs.validators.instance_of(int),
            attrs.validators.gt(0),
        ),
        repr=False,
    )

    x_bins: NDArray[np.float64] = attrs.field(
        default=attrs.Factory(
            partial(_build_histogram2D_bins, axis="x"),
            takes_self=True,
        ),
        converter=np.asarray,
        validator=_validate_x_bins,
        repr=False,
    )
    y_bins: NDArray[np.float64] = attrs.field(
        default=attrs.Factory(
            partial(_build_histogram2D_bins, axis="y"),
            takes_self=True,
        ),
        converter=np.asarray,
        validator=_validate_y_bins,
        repr=False,
    )
    counts: NDArray[np.int64] = attrs.field(
        default=attrs.Factory(_build_zeroed_histogram2D_counts, takes_self=True),
        converter=np.asarray,
        validator=_validate_counts,
        repr=False,
    )

    histogram: NDArray[np.float64] = attrs.field(
        default=attrs.Factory(_build_empty_histogram2D, takes_self=True),
        converter=np.asarray,
        validator=_validate_histogram,
        repr=False,
    )
    realizs: int = attrs.field(
        default=0,
        converter=int,
        validator=attrs.validators.ge(0),
        repr=False,
    )

    def add_histogram_contribution(
        self,
        *,
        x_data: NDArray[np.float64],
        y_data: NDArray[np.float64],
    ) -> None:
        x_indices = np.searchsorted(self.x_bins, x_data, side="right") - 1
        y_indices = np.searchsorted(self.y_bins, y_data, side="right") - 1

        valid = cast(
            NDArray[np.bool_],
            (x_indices >= 0)
            & (x_indices < self.counts.shape[0])
            & (y_indices >= 0)
            & (y_indices < self.counts.shape[1]),
        )

        np.add.at(self.counts, (x_indices[valid], y_indices[valid]), 1)
        object.__setattr__(self, "realizs", self.realizs + 1)

    def compute_histogram(self) -> None:
        total = np.sum(self.counts)
        if total == 0:
            self.histogram.fill(0.0)
            return

        bin_areas = np.outer(np.diff(self.x_bins), np.diff(self.y_bins))
        self.histogram[:] = self.counts / (total * bin_areas)

    def compute_histogram_probabilities(self) -> None:
        total = np.sum(self.counts)
        if total == 0:
            self.histogram.fill(0.0)
            return

        self.histogram[:] = self.counts / total

    def compute_average_x_curve(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        total = np.sum(self.counts)
        if total == 0:
            prob_x_and_y = np.zeros_like(self.histogram)
        else:
            prob_x_and_y = self.counts / total

        prob_x = cast(NDArray[np.float64], np.sum(prob_x_and_y, axis=1))
        prob_y_cnd_x = np.divide(
            prob_x_and_y,
            prob_x[:, None],
            out=np.full_like(prob_x_and_y, np.nan),
            where=prob_x[:, None] > 0,
        )

        y = compute_bin_centers(self.y_bins)
        ave_y_cnd_x = cast(NDArray[np.float64], np.sum(prob_y_cnd_x * y[None, :], axis=1))

        x = compute_bin_centers(self.x_bins)
        return x, ave_y_cnd_x
