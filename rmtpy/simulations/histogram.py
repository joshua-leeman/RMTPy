from __future__ import annotations

import attrs
import numpy as np

import rmtpy.validators
from .data import Data

NUM_BINS_DEFAULT: int = 100
SUPPORT_SCALE_FACTOR_DEFAULT: float = 1.2


def compute_histogram_domain_range(hist: Histogram) -> tuple[float, float]:
    center: float = (hist.support[1] + hist.support[0]) / 2
    radius: float = hist.support_scale_factor * (hist.support[1] - hist.support[0]) / 2
    return tuple([center - radius, center + radius])


def create_histogram_bins(hist: Histogram) -> np.ndarray:
    return np.linspace(-hist.domain_range[0], hist.domain_range[1], hist.num_bins + 1)


def create_zeroed_histogram_counts(hist: Histogram) -> np.ndarray:
    return np.zeros(hist.num_bins, dtype=np.int64)


def create_empty_histogram(hist: Histogram) -> np.ndarray:
    return np.empty(hist.num_bins, dtype=np.float64)


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class Histogram(Data):
    support: tuple[float, float] = attrs.field(
        converter=tuple,
        validator=rmtpy.validators.validate_support,
    )
    support_scale_factor: float = attrs.field(
        default=SUPPORT_SCALE_FACTOR_DEFAULT,
        converter=float,
        validator=attrs.validators.gt(0.0),
        repr=False,
    )
    log_base: float | None = attrs.field(
        default=None,
        converter=attrs.converters.optional(float),
        validator=attrs.validators.optional(attrs.validators.gt(0.0)),
    )

    domain_range: tuple[float, float] = attrs.field(
        default=attrs.Factory(compute_histogram_domain_range, takes_self=True),
        init=False,
        repr=False,
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

    _realizs: int = attrs.field(
        init=False, factory=lambda: np.zeros((1,), dtype=np.int64)
    )

    @property
    def realizs(self) -> int:
        return self._realizs[0]

    def add_histogram_contribution(self, data: np.ndarray) -> None:
        if isinstance(data, (int, float)):
            data = np.array([data], dtype=np.float64)

        indices = np.searchsorted(self.bins, data, side="right") - 1
        valid = (indices >= 0) & (indices < len(self.counts))
        np.add.at(self.counts, indices[valid], 1)
        self._realizs[0] += 1

    def normalize_histogram(self) -> None:
        self.histogram[:] = compute_normalized_histogram(self.bins, self.counts)

    def compute_histogram_as_probabilities(self) -> None:
        self.histogram[:] = self.counts / np.sum(self.counts)


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class Histogram2D(Data):
    x_support: tuple[float, float] = attrs.field(
        converter=tuple,
        validator=rmtpy.validators.validate_support,
    )
    x_support_scale_factor: float = attrs.field(
        default=SUPPORT_SCALE_FACTOR_DEFAULT,
        converter=float,
        validator=attrs.validators.gt(0.0),
        repr=False,
    )
    x_log_base: float | None = attrs.field(
        default=None,
        converter=attrs.converters.optional(float),
        validator=attrs.validators.optional(attrs.validators.gt(0.0)),
    )

    y_support: tuple[float, float] = attrs.field(
        converter=tuple,
        validator=rmtpy.validators.validate_support,
    )
    y_support_scale_factor: float = attrs.field(
        default=SUPPORT_SCALE_FACTOR_DEFAULT,
        converter=float,
        validator=attrs.validators.gt(0.0),
        repr=False,
    )
    y_log_base: float | None = attrs.field(
        default=None,
        converter=attrs.converters.optional(float),
        validator=attrs.validators.optional(attrs.validators.gt(0.0)),
    )

    x_num_bins: int = attrs.field(
        default=NUM_BINS_DEFAULT,
        validator=[
            attrs.validators.instance_of(int),
            attrs.validators.gt(0),
        ],
        repr=False,
    )
    y_num_bins: int = attrs.field(
        default=NUM_BINS_DEFAULT,
        validator=[
            attrs.validators.instance_of(int),
            attrs.validators.gt(0),
        ],
        repr=False,
    )

    x_bins: np.ndarray = attrs.field(init=False, repr=False)

    @x_bins.default
    def _default_x_bins(self) -> np.ndarray:
        return

    y_bins: np.ndarray = attrs.field(init=False, repr=False)

    @y_bins.default
    def _default_y_bins(self) -> np.ndarray:
        return

    counts: np.ndarray = attrs.field(init=False, repr=False)

    @counts.default
    def _default_counts(self) -> np.ndarray:
        return np.zeros((self.x_num_bins, self.y_num_bins), dtype=np.int32)

    histogram: np.ndarray = attrs.field(init=False, repr=False)

    @histogram.default
    def _default_histogram(self) -> np.ndarray:
        return np.empty((self.x_num_bins, self.y_num_bins))

    _realizs: int = attrs.field(
        init=False,
        factory=lambda: np.zeros((1,), dtype=np.int32),
        repr=False,
    )

    @property
    def realizs(self) -> int:
        return self._realizs[0]

    def add_histogram_contribution(
        self, x_data: np.ndarray, y_data: np.ndarray
    ) -> None:
        x_indices = np.searchsorted(self.x_bins, x_data) - 1
        y_indices = np.searchsorted(self.y_bins, y_data) - 1

        valid = (
            (x_indices >= 0)
            & (x_indices < self.counts.shape[0])
            & (y_indices >= 0)
            & (y_indices < self.counts.shape[1])
        )

        np.add.at(self.counts, (x_indices[valid], y_indices[valid]), 1)
        self._realizs[0] += 1

    def compute_histogram_density(self) -> None:
        bin_areas: np.ndarray = np.outer(np.diff(self.x_bins), np.diff(self.y_bins))
        self.histogram[:] = self.counts / (np.sum(self.counts) * bin_areas)

    def compute_histogram_probabilities(self) -> None:
        self.histogram[:] = self.counts / np.sum(self.counts)

    def compute_average_x_curve(self) -> tuple[np.ndarray, np.ndarray]:
        self.compute_histogram_density()
        bin_areas: np.ndarray = np.outer(np.diff(self.x_bins), np.diff(self.y_bins))
        prob_x_and_y: np.ndarray = self.histogram * bin_areas

        prob_x: np.ndarray = np.sum(prob_x_and_y, axis=1)
        prob_y_given_x: np.ndarray = np.divide(
            prob_x_and_y,
            prob_x[:, None],
            out=np.full_like(prob_x_and_y, np.nan),
            where=prob_x[:, None] > 0,
        )

        y_vals: np.ndarray = compute_histogram_bin_centers(
            self.y_bins,
            bins_log_spaced=self.y_log_base is not None,
        )

        average_y_given_x: np.ndarray = np.sum(prob_y_given_x * y_vals[None, :], axis=1)

        x_vals: np.ndarray = compute_histogram_bin_centers(
            self.x_bins,
            bins_log_spaced=self.x_log_base is not None,
        )

        return x_vals, average_y_given_x
