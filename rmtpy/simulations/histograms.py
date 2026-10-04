import math
from functools import partial
from typing import ClassVar, cast, override

import attrs
import numpy as np

from ..density import Support, array_of_floats, compute_bin_centers, compute_histogram
from ..validators import is_support, to_support_pair
from .base_data import Data

NUM_BINS: int = 100

COEFFICIENT_GRID_POLICY: str = "configuration_v1"


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

    @override
    def add_contribution(self, contribution: Data, /) -> None:
        self._validate_contribution(contribution)
        if not isinstance(contribution, Histogram):
            raise TypeError("Histogram contribution is malformed.")
        if (
            contribution.support != self.support
            or contribution.log_base != self.log_base
            or contribution.num_bins != self.num_bins
            or not np.array_equal(contribution.bins, self.bins)
        ):
            raise ValueError(
                f"Histogram contribution `{self._file_name}` has an incompatible grid."
            )

        self.counts[:] += contribution.counts
        self._add_realizations(contribution)

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

    @override
    def compute_statistics(self) -> None:
        self.compute_histogram()


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class CoefficientsHistogram(Histogram):
    coefficient_name: ClassVar[str]

    underflow: int = attrs.field(
        default=0,
        converter=int,
        validator=attrs.validators.ge(0),
        metadata={"archive_optional": True},
        repr=False,
    )
    overflow: int = attrs.field(
        default=0,
        converter=int,
        validator=attrs.validators.ge(0),
        metadata={"archive_optional": True},
        repr=False,
    )

    @classmethod
    def create[HistogramType: CoefficientsHistogram](
        cls: type[HistogramType],
        *,
        degree: int,
        dimension: int,
    ) -> HistogramType:
        degree = int(degree)
        dimension = int(dimension)
        if degree < 1:
            raise ValueError("Coefficient polynomial degree must be positive.")
        if dimension < 1:
            raise ValueError("Coefficient histogram dimension must be positive.")

        half_width = (degree + 1) / math.sqrt(dimension)
        num_bins = max(NUM_BINS, math.ceil(math.sqrt(dimension)))
        histogram = cls(
            _file_name=f"{cls.coefficient_name}_coeff_{degree}_histogram",
            support=(-half_width, half_width),
            num_bins=num_bins,
        )
        histogram.attach_metadata(
            {
                "degree": degree,
                "unfolding": "raw",
                "grid_policy": COEFFICIENT_GRID_POLICY,
                "dimension": dimension,
            }
        )
        return histogram

    @override
    def add_histogram_contribution(
        self,
        data: np.ndarray[tuple[int], np.dtype[np.floating]],
        /,
    ) -> None:
        values = np.asarray(data)
        if not np.all(np.isfinite(values)):
            raise ValueError("Coefficient samples must be finite.")

        object.__setattr__(
            self,
            "underflow",
            self.underflow
            + int(np.count_nonzero(values < cast(np.floating, self.bins[0]))),
        )
        object.__setattr__(
            self,
            "overflow",
            self.overflow
            + int(np.count_nonzero(values >= cast(np.floating, self.bins[-1]))),
        )
        super().add_histogram_contribution(values)

    @override
    def add_contribution(self, contribution: Data, /) -> None:
        if not isinstance(contribution, CoefficientsHistogram):
            raise TypeError("Coefficient histogram contribution is malformed.")
        if contribution.metadata.get("grid_policy") != COEFFICIENT_GRID_POLICY:
            raise ValueError(
                f"Coefficient histogram `{contribution._file_name}` predates the "
                + "deterministic aggregation grid and cannot be aggregated exactly."
            )

        super().add_contribution(contribution)
        object.__setattr__(self, "underflow", self.underflow + contribution.underflow)
        object.__setattr__(self, "overflow", self.overflow + contribution.overflow)


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class MeanScaledHistogram(Histogram):
    sample_sum: float | None = attrs.field(
        default=None,
        converter=attrs.converters.optional(float),
        validator=attrs.validators.optional(attrs.validators.ge(0.0)),
        metadata={"archive_optional": True},
        repr=False,
    )

    def __attrs_post_init__(self) -> None:
        if self.sample_sum is not None:
            return

        average = self.metadata.get("average_width", 0.0)
        recovered_sum = (
            float(average) * self.realizs
            if isinstance(average, int | float) and self.realizs > 0
            else 0.0
        )
        object.__setattr__(self, "sample_sum", recovered_sum)

    @override
    def _aggregation_metadata(self) -> dict[str, object]:
        return {
            key: value for key, value in self.metadata.items() if key != "average_width"
        }

    def _canonical_bins(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        return array_of_floats(
            support=self.support,
            num_pts=self.num_bins + 1,
            log_base=self.log_base,
        )

    @override
    def add_histogram_contribution(
        self,
        data: np.ndarray[tuple[int], np.dtype[np.floating]],
        /,
    ) -> None:
        values = np.asarray(data)
        super().add_histogram_contribution(values)
        sample_sum = cast(float, self.sample_sum)
        object.__setattr__(self, "sample_sum", sample_sum + float(np.sum(values)))

    @override
    def add_contribution(self, contribution: Data, /) -> None:
        self._validate_contribution(contribution)
        if not isinstance(contribution, MeanScaledHistogram):
            raise TypeError("Mean-scaled histogram contribution is malformed.")
        if (
            contribution.support != self.support
            or contribution.log_base != self.log_base
            or contribution.num_bins != self.num_bins
        ):
            raise ValueError(
                f"Histogram contribution `{self._file_name}` has an incompatible grid."
            )

        contribution_sum = cast(float, contribution.sample_sum)
        if contribution.realizs > 0:
            if not np.isfinite(contribution_sum) or contribution_sum <= 0.0:
                raise ValueError(
                    f"Histogram contribution `{self._file_name}` has an invalid "
                    + "sample sum."
                )
            average = contribution_sum / contribution.realizs
            expected_bins = contribution._canonical_bins() / average
        else:
            if contribution_sum != 0.0:
                raise ValueError(
                    f"Histogram contribution `{self._file_name}` has an invalid "
                    + "sample sum."
                )
            expected_bins = contribution._canonical_bins()
        if not np.allclose(contribution.bins, expected_bins, rtol=1e-12, atol=0.0):
            raise ValueError(
                f"Histogram contribution `{self._file_name}` has malformed scaled bins."
            )

        self.counts[:] += contribution.counts
        object.__setattr__(
            self,
            "sample_sum",
            cast(float, self.sample_sum) + contribution_sum,
        )
        self._add_realizations(contribution)

    @override
    def compute_statistics(self) -> None:
        if self.realizs == 0:
            raise ValueError("A mean-scaled histogram requires realizations.")

        average = cast(float, self.sample_sum) / self.realizs
        if not np.isfinite(average) or average <= 0.0:
            index = self.metadata.get("index")
            raise ValueError(
                f"Average width for index {index} must be positive and finite."
            )

        self.attach_metadata({"average_width": average})
        self.bins[:] = self._canonical_bins() / average
        self.compute_histogram()


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

    @override
    def add_contribution(self, contribution: Data, /) -> None:
        self._validate_contribution(contribution)
        if not isinstance(contribution, Histogram2D):
            raise TypeError("2-D histogram contribution is malformed.")
        if (
            contribution.x_support != self.x_support
            or contribution.y_support != self.y_support
            or contribution.x_log_base != self.x_log_base
            or contribution.y_log_base != self.y_log_base
            or contribution.x_num_bins != self.x_num_bins
            or contribution.y_num_bins != self.y_num_bins
            or not np.array_equal(contribution.x_bins, self.x_bins)
            or not np.array_equal(contribution.y_bins, self.y_bins)
        ):
            raise ValueError(
                f"2-D histogram contribution `{self._file_name}` has an "
                + "incompatible grid."
            )

        self.counts[:] += contribution.counts
        self._add_realizations(contribution)

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

    @override
    def compute_statistics(self) -> None:
        self.compute_histogram()

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
