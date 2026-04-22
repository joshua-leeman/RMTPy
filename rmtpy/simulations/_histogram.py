from __future__ import annotations

import numpy as np
from attrs import frozen, field
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

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
        return np.zeros(self.num_bins, dtype=np.int32)

    histogram: np.ndarray = field(init=False)

    @histogram.default
    def _default_histogram(self) -> np.ndarray:
        return np.empty(self.num_bins, np.float64)

    _numerical_pdf: PchipInterpolator | None = field(init=False, default=None)
    _numerical_cdf: PchipInterpolator | None = field(init=False, default=None)
    _realizs_count: int = field(
        init=False, factory=lambda: np.zeros((1,), dtype=np.int32)
    )

    @property
    def realizs(self) -> int:
        return self._realizs_count[0]

    def add_histogram_contribution(self, data: np.ndarray) -> None:
        indices = np.searchsorted(self.bins, data, side="right") - 1
        valid = (indices >= 0) & (indices < len(self.counts))
        np.add.at(self.counts, indices[valid], 1)
        self._realizs_count[0] += 1

    def compute_histogram_density(self) -> None:
        self.histogram[:] = self.counts / (np.sum(self.counts) * np.diff(self.bins))

    def compute_histogram_probabilities(self) -> None:
        self.histogram[:] = self.counts / np.sum(self.counts)

    def numerical_pdf(
        self, values: int | float | np.ndarray, _sigma: float = 1.5
    ) -> np.ndarray:
        if isinstance(values, (int, float)):
            values: np.ndarray = np.array([values], dtype=np.float64)

        if self._numerical_pdf is None:
            object.__setattr__(
                self,
                "_numerical_pdf",
                self._create_numerical_pdf(sigma=_sigma),
            )

        return self._numerical_pdf(values)

    def numerical_cdf(
        self, values: int | float | np.ndarray, _sigma: float = 1.5
    ) -> np.ndarray:
        if isinstance(values, (int, float)):
            values: np.ndarray = np.array([values], dtype=np.float64)

        if self._numerical_cdf is None:
            object.__setattr__(
                self,
                "_numerical_cdf",
                self._create_numerical_cdf(sigma=_sigma),
            )

        return self._numerical_cdf(values)

    def unfold(self, dimension: int, values: np.ndarray) -> np.ndarray:
        return dimension * (
            self.numerical_cdf(values) - self.numerical_cdf(np.array([0.0]))
        )

    def unfold_widths(
        self, dimension: int, values: np.ndarray, widths: np.ndarray
    ) -> np.ndarray:
        return dimension * (
            self.numerical_cdf(values + widths / 2)
            - self.numerical_cdf(values - widths / 2)
        )

    def _create_numerical_pdf(self, sigma: float = 2.0) -> PchipInterpolator:
        self.compute_histogram_density()
        centers: np.ndarray = (self.bins[:-1] + self.bins[1:]) / 2
        pdf_vals: np.ndarray = gaussian_filter1d(self.histogram, sigma=sigma)

        return PchipInterpolator(centers, pdf_vals, extrapolate=True)

    def _create_numerical_cdf(self, sigma: float = 2.0) -> PchipInterpolator:
        self.compute_histogram_density()
        centers: np.ndarray = (self.bins[:-1] + self.bins[1:]) / 2
        pdf_vals: np.ndarray = self.numerical_pdf(centers, sigma=sigma)
        cdf_vals: np.ndarray = cumulative_trapezoid(pdf_vals, centers, initial=0)

        return PchipInterpolator(centers, cdf_vals, extrapolate=True)


@frozen(kw_only=True, repr=False, eq=False, weakref_slot=False, getstate_setstate=False)
class Histogram2D(Data):
    x_support: tuple[float, float]
    y_support: tuple[float, float]
    x_scale: float = 1.0
    y_scale: float = 1.0
    x_num_bins: int = 100
    y_num_bins: int = 100

    x_bins: np.ndarray = field(init=False)

    @x_bins.default
    def _default_x_bins(self) -> np.ndarray:
        return self.x_scale * np.linspace(
            self.x_support[0], self.x_support[1], self.x_num_bins + 1
        )

    y_bins: np.ndarray = field(init=False)

    @y_bins.default
    def _default_y_bins(self) -> np.ndarray:
        return self.y_scale * np.linspace(
            self.y_support[0], self.y_support[1], self.y_num_bins + 1
        )

    counts: np.ndarray = field(init=False)

    @counts.default
    def _default_counts(self) -> np.ndarray:
        return np.zeros((self.x_num_bins, self.y_num_bins), dtype=np.int32)

    histogram: np.ndarray = field(init=False)

    @histogram.default
    def _default_histogram(self) -> np.ndarray:
        return np.empty((self.x_num_bins, self.y_num_bins))

    _numerical_pdf: PchipInterpolator | None = field(init=False, default=None)
    _numerical_cdf: PchipInterpolator | None = field(init=False, default=None)
    _realizs_count: int = field(
        init=False, factory=lambda: np.zeros((1,), dtype=np.int32)
    )

    @property
    def realizs(self) -> int:
        return self._realizs_count[0]

    def add_histogram_contribution(
        self, x_data: np.ndarray, y_data: np.ndarray
    ) -> None:
        x_indices = np.searchsorted(self.x_bins, x_data) - 1
        y_indices = np.searchsorted(self.y_bins, y_data) - 1

        np.add.at(self.counts, (x_indices, y_indices), 1)
        self._realizs_count[0] += 1

    def compute_histogram_density(self) -> None:
        bin_areas: np.ndarray = (
            np.diff(self.x_bins)[:, None] * np.diff(self.y_bins)[None, :]
        )
        self.histogram[:] = self.counts / (np.sum(self.counts) * bin_areas)

    def compute_histogram_probabilities(self) -> None:
        self.histogram[:] = self.counts / np.sum(self.counts)

    def numerical_pdf(
        self, values: int | float | np.ndarray, _sigma: float = 1.5
    ) -> np.ndarray:
        if isinstance(values, (int, float)):
            values: np.ndarray = np.array([values], dtype=np.float64)

        if self._numerical_pdf is None:
            object.__setattr__(
                self,
                "_numerical_pdf",
                self._create_numerical_pdf(sigma=_sigma),
            )

        return self._numerical_pdf(values)

    def numerical_cdf(
        self, values: int | float | np.ndarray, _sigma: float = 1.5
    ) -> np.ndarray:
        if isinstance(values, (int, float)):
            values: np.ndarray = np.array([values], dtype=np.float64)

        if self._numerical_cdf is None:
            object.__setattr__(
                self,
                "_numerical_cdf",
                self._create_numerical_cdf(sigma=_sigma),
            )

        return self._numerical_cdf(values)

    def unfold(self, dimension: int, values: np.ndarray) -> np.ndarray:
        return dimension * (
            self.numerical_cdf(values) - self.numerical_cdf(np.array([0.0]))
        )

    def unfold_widths(
        self, dimension: int, values: np.ndarray, widths: np.ndarray
    ) -> np.ndarray:
        return dimension * (
            self.numerical_cdf(values + widths / 2)
            - self.numerical_cdf(values - widths / 2)
        )

    def compute_average_curve(self) -> tuple[np.ndarray, np.ndarray]:
        self.compute_histogram_density()
        bin_areas: np.ndarray = (
            np.diff(self.x_bins)[:, None] * np.diff(self.y_bins)[None, :]
        )
        prob_x_and_y: np.ndarray = self.histogram * bin_areas

        prob_x: np.ndarray = np.sum(prob_x_and_y, axis=1)
        prob_y_given_x: np.ndarray = np.divide(
            prob_x_and_y,
            prob_x[:, None],
            out=np.full_like(prob_x_and_y, np.nan),
            where=prob_x[:, None] > 0,
        )

        y_vals: np.ndarray = np.sqrt(self.y_bins[:-1] * self.y_bins[1:])
        ave_y_given_x: np.ndarray = np.sum(prob_y_given_x * y_vals[None, :], axis=1)

        x_vals: np.ndarray = (self.x_bins[:-1] + self.x_bins[1:]) / 2
        return x_vals, ave_y_given_x

    def _create_numerical_pdf(self, sigma: float = 2.0) -> PchipInterpolator:
        self.compute_histogram_density()
        centers: np.ndarray = (self.x_bins[:-1] + self.x_bins[1:]) / 2
        pdf_vals: np.ndarray = gaussian_filter1d(self.histogram, sigma=sigma)

        return PchipInterpolator(centers, pdf_vals, extrapolate=True)

    def _create_numerical_cdf(self, sigma: float = 2.0) -> PchipInterpolator:
        self.compute_histogram_density()
        centers: np.ndarray = (self.x_bins[:-1] + self.x_bins[1:]) / 2
        pdf_vals: np.ndarray = self.numerical_pdf(centers, sigma=sigma)
        cdf_vals: np.ndarray = cumulative_trapezoid(pdf_vals, centers, initial=0)

        return PchipInterpolator(centers, cdf_vals, extrapolate=True)
