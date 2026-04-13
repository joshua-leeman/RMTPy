from __future__ import annotations

from functools import lru_cache

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
    num_bins: int = 150

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

    _numerical_pdf: PchipInterpolator | None = field(init=False, default=None)
    _numerical_cdf: PchipInterpolator | None = field(init=False, default=None)
    _realizs_count: int = field(init=False, factory=lambda: np.zeros((1,), dtype=int))

    @property
    def realizs(self) -> int:
        return self._realizs_count[0]

    def add_histogram_contribution(self, counts: np.ndarray) -> None:
        self.counts[:] += np.histogram(counts, bins=self.bins)[0]
        self._realizs_count[0] += 1

    def compute_histogram(self) -> None:
        self.histogram[:] = self.counts / np.sum(self.counts * np.diff(self.bins))

    def compute_histogram_with_normalization(self, normalization: float) -> None:
        self.histogram[:] = self.counts / normalization

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

    def unfold_locally(
        self, dimension: int, resonances: np.ndarray, values: np.ndarray
    ) -> np.ndarray:
        return dimension * (
            self.numerical_cdf(resonances + values / 2)
            - self.numerical_cdf(resonances - values / 2)
        )

    def _create_numerical_pdf(self, sigma: float = 2.0) -> PchipInterpolator:
        self.compute_histogram()
        centers: np.ndarray = (self.bins[:-1] + self.bins[1:]) / 2
        pdf_vals: np.ndarray = gaussian_filter1d(self.histogram, sigma=sigma)

        return PchipInterpolator(centers, pdf_vals, extrapolate=True)

    def _create_numerical_cdf(self, sigma: float = 2.0) -> PchipInterpolator:
        self.compute_histogram()
        centers: np.ndarray = (self.bins[:-1] + self.bins[1:]) / 2
        pdf_vals: np.ndarray = self.numerical_pdf(centers, sigma=sigma)
        cdf_vals: np.ndarray = cumulative_trapezoid(pdf_vals, centers, initial=0)

        return PchipInterpolator(centers, cdf_vals, extrapolate=True)
