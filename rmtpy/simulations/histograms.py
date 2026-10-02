from functools import partial

import attrs
import numpy as np

from ..density import Support, array_of_floats, compute_bin_centers, compute_histogram
from ..validators import is_support, to_support_pair
from .base_data import Data

NUM_BINS: int = 100


def _build_empty_histogram(
    histogram: Histogram,
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    return np.zeros(histogram.num_bins, dtype=np.float64)


def _build_histogram_bins(
    histogram: Histogram,
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    return array_of_floats(
        support=histogram.support,
        num_pts=histogram.num_bins + 1,
        log_base=histogram.log_base,
    )


def _build_zeroed_histogram_counts(
    histogram: Histogram,
) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
    return np.zeros(histogram.num_bins, dtype=np.int64)


def _validate_histogram_bins(
    histogram: Histogram,
    _: object,
    values: np.ndarray[tuple[int], np.dtype[np.floating]],
) -> None:
    if (
        values.shape != (histogram.num_bins + 1,)
        or not np.issubdtype(values.dtype, np.floating)
        or not np.all(np.isfinite(values))
        or np.any(np.diff(values) <= 0.0)
    ):
        raise ValueError("Histogram bins do not match the declared bin count.")


def _validate_histogram_counts(
    histogram: Histogram,
    _: object,
    values: np.ndarray[tuple[int], np.dtype[np.int64]],
) -> None:
    if (
        values.shape != (histogram.num_bins,)
        or not np.issubdtype(values.dtype, np.integer)
        or np.any(values < 0)
    ):
        raise ValueError("Histogram counts do not match the declared bin count.")


def _validate_histogram(
    histogram: Histogram,
    _: object,
    values: np.ndarray[tuple[int], np.dtype[np.floating]],
) -> None:
    if (
        values.shape != (histogram.num_bins,)
        or not np.issubdtype(values.dtype, np.floating)
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
    ):
        raise ValueError("Histogram values do not match the declared bin count.")


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class Histogram(Data):
    support: Support = attrs.field(
        converter=to_support_pair,
        validator=is_support,
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

    bins: np.ndarray[tuple[int], np.dtype[np.floating]] = attrs.field(
        default=attrs.Factory(_build_histogram_bins, takes_self=True),
        converter=np.asarray,
        validator=_validate_histogram_bins,
        repr=False,
    )
    counts: np.ndarray[tuple[int], np.dtype[np.int64]] = attrs.field(
        default=attrs.Factory(_build_zeroed_histogram_counts, takes_self=True),
        converter=np.asarray,
        validator=_validate_histogram_counts,
        repr=False,
    )
    histogram: np.ndarray[tuple[int], np.dtype[np.floating]] = attrs.field(
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

    def add_histogram_contribution(
        self,
        data: np.ndarray[tuple[int], np.dtype[np.floating]],
        /,
    ) -> None:
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


def _build_empty_histogram2D(
    histogram: Histogram2D,
) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
    return np.zeros((histogram.x_num_bins, histogram.y_num_bins), dtype=np.float64)


def _build_histogram2D_axis_bins(
    histogram: Histogram2D,
    axis: str,
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    if axis.strip().lower() == "x":
        return array_of_floats(
            support=histogram.x_support,
            num_pts=histogram.x_num_bins + 1,
            log_base=histogram.x_log_base,
        )

    elif axis.strip().lower() == "y":
        return array_of_floats(
            support=histogram.y_support,
            num_pts=histogram.y_num_bins + 1,
            log_base=histogram.y_log_base,
        )

    else:
        raise ValueError("`axis` must be either 'x' and 'y'.")


def _build_zeroed_histogram2D_counts(
    histogram: Histogram2D,
) -> np.ndarray[tuple[int, int], np.dtype[np.int64]]:
    return np.zeros((histogram.x_num_bins, histogram.y_num_bins), dtype=np.int64)


def _validate_histogram2D_x_bins(
    histogram: Histogram2D,
    _: object,
    values: np.ndarray[tuple[int], np.dtype[np.floating]],
) -> None:
    if (
        values.shape != (histogram.x_num_bins + 1,)
        or not np.issubdtype(values.dtype, np.floating)
        or not np.all(np.isfinite(values))
        or np.any(np.diff(values) <= 0.0)
    ):
        raise ValueError("2-D histogram x bins do not match the declared bin count.")


def _validate_histogram2D_y_bins(
    histogram: Histogram2D,
    _: object,
    values: np.ndarray[tuple[int], np.dtype[np.floating]],
) -> None:
    if (
        values.shape != (histogram.y_num_bins + 1,)
        or not np.issubdtype(values.dtype, np.floating)
        or not np.all(np.isfinite(values))
        or np.any(np.diff(values) <= 0.0)
    ):
        raise ValueError("2-D histogram y bins do not match the declared bin count.")


def _validate_histogram2D_counts(
    histogram: Histogram2D,
    _: object,
    values: np.ndarray[tuple[int, int], np.dtype[np.int64]],
) -> None:
    if (
        values.shape != (histogram.x_num_bins, histogram.y_num_bins)
        or not np.issubdtype(values.dtype, np.integer)
        or np.any(values < 0)
    ):
        raise ValueError("2-D histogram counts do not match declared bin counts.")


def _validate_histogram2D(
    histogram: Histogram2D,
    _: object,
    values: np.ndarray[tuple[int, int], np.dtype[np.floating]],
) -> None:
    if (
        values.shape != (histogram.x_num_bins, histogram.y_num_bins)
        or not np.issubdtype(values.dtype, np.floating)
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
    ):
        raise ValueError("2-D histogram values do not match declared bin counts.")


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class Histogram2D(Data):
    x_support: Support = attrs.field(
        converter=to_support_pair,
        validator=is_support,
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

    y_support: Support = attrs.field(
        converter=to_support_pair,
        validator=is_support,
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

    x_bins: np.ndarray[tuple[int], np.dtype[np.floating]] = attrs.field(
        default=attrs.Factory(
            partial(_build_histogram2D_axis_bins, axis="x"),
            takes_self=True,
        ),
        converter=np.asarray,
        validator=_validate_histogram2D_x_bins,
        repr=False,
    )
    y_bins: np.ndarray[tuple[int], np.dtype[np.floating]] = attrs.field(
        default=attrs.Factory(
            partial(_build_histogram2D_axis_bins, axis="y"),
            takes_self=True,
        ),
        converter=np.asarray,
        validator=_validate_histogram2D_y_bins,
        repr=False,
    )
    counts: np.ndarray[tuple[int, int], np.dtype[np.int64]] = attrs.field(
        default=attrs.Factory(_build_zeroed_histogram2D_counts, takes_self=True),
        converter=np.asarray,
        validator=_validate_histogram2D_counts,
        repr=False,
    )

    histogram: np.ndarray[tuple[int, int], np.dtype[np.floating]] = attrs.field(
        default=attrs.Factory(_build_empty_histogram2D, takes_self=True),
        converter=np.asarray,
        validator=_validate_histogram2D,
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
        x_data: np.ndarray[tuple[int], np.dtype[np.floating]],
        y_data: np.ndarray[tuple[int], np.dtype[np.floating]],
    ) -> None:
        x_indices = np.searchsorted(self.x_bins, x_data, side="right") - 1
        y_indices = np.searchsorted(self.y_bins, y_data, side="right") - 1

        valid = np.asarray(
            (x_indices >= 0)
            & (x_indices < self.counts.shape[0])
            & (y_indices >= 0)
            & (y_indices < self.counts.shape[1]),
            dtype=np.bool_,
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

    def compute_average_x_curve(
        self,
    ) -> tuple[
        np.ndarray[tuple[int], np.dtype[np.floating]],
        np.ndarray[tuple[int], np.dtype[np.floating]],
    ]:
        total = np.sum(self.counts)
        if total == 0:
            prob_x_and_y = np.zeros_like(self.histogram)
        else:
            prob_x_and_y = self.counts / total

        prob_x = np.asarray(np.sum(prob_x_and_y, axis=1), dtype=np.float64)
        prob_y_cnd_x = np.divide(
            prob_x_and_y,
            prob_x[:, None],
            out=np.full_like(prob_x_and_y, np.nan),
            where=prob_x[:, None] > 0,
        )

        y = compute_bin_centers(self.y_bins)
        ave_y_cnd_x = np.asarray(
            np.sum(prob_y_cnd_x * y[None, :], axis=1),
            dtype=np.float64,
        )

        x = compute_bin_centers(self.x_bins)
        return x, ave_y_cnd_x
