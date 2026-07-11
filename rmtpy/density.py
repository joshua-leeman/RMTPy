from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TypeAlias

import attrs
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

import rmtpy.validators

MAX_POLYNOMIAL_DEGREE: int = 10

SUPPORT_SCALE_FACTOR: float = 1.2

NUM_REALIZATIONS_MIN: int = 10

NUM_HISTOGRAM_COUNTS: int = 2**13

GAUSSIAN_KERNEL_STANDARD_DEVIATION: float = 2.0

NUM_POINTS: int = 1000

Support: TypeAlias = tuple[float, float]
ArrayFunction: TypeAlias = Callable[[np.ndarray], np.ndarray]
PolynomialFunction: TypeAlias = Callable[[np.ndarray, int], np.ndarray]


def array_of_floats(
    *, support: Support, num_pts: int, log_base: float | None = None
) -> np.ndarray:
    rmtpy.validators.validate_support(support)
    if log_base is None:
        return np.linspace(*support, num_pts)
    else:
        return np.logspace(*support, num_pts, base=log_base)


def compute_bin_centers(bins: np.ndarray) -> np.ndarray:
    bins = np.asarray(bins)
    if bins.ndim != 1:
        raise ValueError("`bins` must be one-dimensional.")
    if len(bins) < 2:
        raise ValueError("At least two histogram bin edges are required.")
    if np.any(np.diff(bins) <= 0):
        raise ValueError("`bins` must be strictly increasing.")

    if np.all(bins > 0.0) and np.allclose(bins[1:] / bins[:-1], bins[1] / bins[0]):
        return np.sqrt(bins[:-1] * bins[1:])

    return (bins[:-1] + bins[1:]) / 2


def compute_histogram(counts: np.ndarray, *, bins: np.ndarray) -> np.ndarray:
    counts, bins = np.asarray(counts), np.asarray(bins)
    if bins.ndim != 1 or counts.ndim != 1:
        raise ValueError("`bins` and `counts` must be one-dimensional.")
    if len(bins) != len(counts) + 1:
        raise ValueError("`bins` must have exactly one more entry than `counts`.")
    if np.any(np.diff(bins) <= 0):
        raise ValueError("`bins` must be strictly increasing.")
    if np.any(counts < 0):
        raise ValueError("`counts` must be non-negative.")

    total_counts = np.sum(counts)
    if total_counts == 0:
        raise ValueError("Cannot normalize histogram with zero total counts.")

    return counts / (total_counts * np.diff(bins))


def create_pdf_interpolator_from_histogram(
    histogram: np.ndarray,
    *,
    bins: np.ndarray,
    kernel_std_dev: float = GAUSSIAN_KERNEL_STANDARD_DEVIATION,
) -> PchipInterpolator:
    histogram, bins = np.asarray(histogram), np.asarray(bins)

    centers = compute_bin_centers(bins)
    pdf_values = gaussian_filter1d(histogram, kernel_std_dev)

    return PchipInterpolator(centers, pdf_values, extrapolate=True)


def create_cdf_interpolator_from_pdf(
    pdf: Callable[[np.ndarray], np.ndarray],
    *,
    inputs: np.ndarray,
    left_tail_mass: float = 0.0,
) -> PchipInterpolator:
    inputs = np.asarray(inputs)
    if inputs.ndim != 1:
        raise ValueError("CDF interpolation `inputs` must be one-dimensional.")
    if len(inputs) < 2:
        raise ValueError("CDF interpolation requires at least two entries in `inputs`.")
    if not np.all(np.isfinite(inputs)):
        raise ValueError("CDF interpolation `inputs` must be finite.")
    if np.any(np.diff(inputs) <= 0):
        raise ValueError("CDF interpolation `inputs` must be strictly increasing.")
    if not np.isfinite(left_tail_mass) or left_tail_mass < 0.0:
        raise ValueError("`left_tail_mass` must be a finite non-negative number.")

    cdf_values = left_tail_mass + cumulative_trapezoid(pdf(inputs), inputs, initial=0.0)

    return PchipInterpolator(inputs, cdf_values, extrapolate=True)


def unfold_values_with_cdf(
    values: np.ndarray,
    *,
    cdf: Callable[[np.ndarray], np.ndarray] | None = None,
    dimension: int | None = None,
) -> np.ndarray:
    values = np.asarray(values)

    if cdf is None:
        return values

    return dimension * (cdf(values) - cdf(np.array([0.0])))


def unfold_widths_with_cdf(
    *,
    widths: np.ndarray,
    centers: np.ndarray,
    cdf: Callable[[np.ndarray], np.ndarray] | None,
    dimension: int,
) -> np.ndarray:
    centers, widths = np.asarray(centers), np.asarray(widths)

    if cdf is None:
        return widths

    return dimension * (cdf(centers + widths / 2) - cdf(centers - widths / 2))


def compute_default_number_of_bins(density: DensityModel) -> int:
    return int(np.ceil(np.sqrt(density.dimension)))


def compute_optimal_realizations(density: DensityModel) -> int:
    return max(NUM_HISTOGRAM_COUNTS // density.dimension, NUM_REALIZATIONS_MIN)


def is_polynomial_expansion_completely_provided(
    density: DensityModel, _, weight_function: ArrayFunction | None
) -> None:
    if (weight_function is None) != (density.polynomials is None):
        raise ValueError(
            "A polynomial expansion requires both a set of orthogonal "
            "polynomials and its associated weight function."
        )


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class DensityModel:
    dimension: int = attrs.field(
        converter=int,
        validator=attrs.validators.gt(0),
    )
    support: Support = attrs.field(
        converter=tuple,
        validator=lambda _, __, support: rmtpy.validators.validate_support(support),
    )
    polynomials: PolynomialFunction | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(attrs.validators.is_callable()),
    )
    max_polynomial_degree: int = attrs.field(
        default=MAX_POLYNOMIAL_DEGREE,
        converter=int,
        validator=attrs.validators.ge(0),
    )
    weight_function: ArrayFunction | None = attrs.field(
        default=None,
        validator=[
            attrs.validators.optional(attrs.validators.is_callable()),
            is_polynomial_expansion_completely_provided,
        ],
    )
    sample_stream: Callable[[int, bool], Iterator[np.ndarray]] = attrs.field(
        validator=attrs.validators.is_callable(),
        repr=False,
    )

    support_scale_factor: float = attrs.field(
        default=SUPPORT_SCALE_FACTOR,
        converter=float,
        repr=False,
    )
    kernel_std_dev: float = attrs.field(
        default=GAUSSIAN_KERNEL_STANDARD_DEVIATION,
        converter=float,
        validator=attrs.validators.gt(0.0),
        repr=False,
    )
    num_pts: int = attrs.field(
        default=NUM_POINTS,
        converter=int,
        validator=attrs.validators.gt(0),
        repr=False,
    )
    num_bins: int = attrs.field(
        default=attrs.Factory(compute_default_number_of_bins, takes_self=True),
        converter=int,
        validator=attrs.validators.gt(0),
        repr=False,
    )
    optimal_realizs: int = attrs.field(
        default=attrs.Factory(compute_optimal_realizations, takes_self=True),
        init=False,
        repr=False,
    )

    _average_coeffs: np.ndarray | None = attrs.field(
        default=None,
        init=False,
        repr=False,
    )
    _average_pdf_interpolator: PchipInterpolator | None = attrs.field(
        default=None,
        init=False,
        repr=False,
    )
    _average_cdf_interpolator: PchipInterpolator | None = attrs.field(
        default=None,
        init=False,
        repr=False,
    )
    _weight_cdf_interpolator: PchipInterpolator | None = attrs.field(
        default=None,
        init=False,
        repr=False,
    )

    @property
    def has_polynomial_expansion(self) -> bool:
        return self.polynomials is not None and self.weight_function is not None

    @property
    def average_coeffs(self) -> np.ndarray | None:
        if self.has_polynomial_expansion and self._average_coeffs is None:
            average_coeffs = self._compute_average_coeffs()
            object.__setattr__(self, "_average_coeffs", average_coeffs)

        return self._average_coeffs

    @property
    def support_radius(self) -> float:
        return (self.support[1] - self.support[0]) / 2

    @property
    def plot_range(self) -> Support:
        center = sum(self.support) / 2
        radius = self.support_scale_factor * self.support_radius
        return center - radius, center + radius

    def average_cdf(self, points: np.ndarray) -> np.ndarray:
        if not self.has_polynomial_expansion:
            return self._average_cdf_from_samples(np.asarray(points))

        return self._average_cdf_from_polynomials(np.asarray(points))

    def average_pdf(self, points: np.ndarray) -> np.ndarray:
        if not self.has_polynomial_expansion:
            return self._average_pdf_from_samples(np.asarray(points))

        return self._average_pdf_from_polynomials(np.asarray(points))

    def compute_polynomial_weight(self, inputs: np.ndarray) -> np.ndarray:
        if not self.has_polynomial_expansion:
            raise NotImplementedError()

        return self.weight_function(np.asarray(inputs))

    def compute_polynomials(self, inputs: np.ndarray) -> np.ndarray:
        if not self.has_polynomial_expansion:
            raise NotImplementedError()

        center = sum(self.support) / 2
        x = (np.asarray(inputs) - center) / self.support_radius

        return self.polynomials(x, self.max_polynomial_degree)

    def compute_variate_coeffs(self, sample: np.ndarray) -> np.ndarray:
        polynomials = self.compute_polynomials(np.asarray(sample))
        return np.mean(polynomials, axis=1)

    def create_variate_cdf_interpolator(
        self,
        *,
        interval: Support | None = None,
        coeffs: np.ndarray | None = None,
        sample: np.ndarray | None = None,
    ) -> PchipInterpolator:
        if interval is None:
            interval = self.plot_range
        else:
            rmtpy.validators.validate_support(interval)

        def pdf(points: np.ndarray) -> np.ndarray:
            return self.variate_pdf(points, coeffs=coeffs, sample=sample)

        if interval[0] > self.plot_range[0]:
            left_tail_support = (self.plot_range[0], interval[0])
            left_tail = array_of_floats(support=left_tail_support, num_pts=self.num_pts)
            left_tail_mass = cumulative_trapezoid(pdf(left_tail), left_tail)[-1]
        else:
            left_tail_mass = 0.0

        return create_cdf_interpolator_from_pdf(
            pdf,
            inputs=array_of_floats(support=interval, num_pts=self.num_pts),
            left_tail_mass=left_tail_mass,
        )

    def variate_pdf(
        self,
        points: np.ndarray,
        *,
        coeffs: np.ndarray | None = None,
        sample: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.has_polynomial_expansion:
            return self._variate_pdf_from_polynomials(
                points, coeffs=coeffs, sample=sample
            )

        return self._variate_pdf_from_sample(points, sample=sample)

    def variate_cdf(
        self,
        points: np.ndarray,
        *,
        coeffs: np.ndarray | None = None,
        sample: np.ndarray | None = None,
    ) -> np.ndarray:
        cdf_interpolator = self.create_variate_cdf_interpolator(
            coeffs=coeffs, sample=sample
        )
        return cdf_interpolator(points)

    def weight_pdf(self, points: np.ndarray) -> np.ndarray:
        if not self.has_polynomial_expansion:
            return self._average_pdf_from_samples(points)

        return self.compute_polynomial_weight(points)

    def weight_cdf(self, points: np.ndarray) -> np.ndarray:
        if not self.has_polynomial_expansion:
            return self._average_cdf_from_samples(points)

        if self._weight_cdf_interpolator is None:
            _weight_cdf = self._create_weight_cdf_interpolator()
            object.__setattr__(self, "_weight_cdf_interpolator", _weight_cdf)

        return self._weight_cdf_interpolator(points)

    def _average_cdf_from_polynomials(self, points: np.ndarray) -> np.ndarray:
        return self.create_variate_cdf_interpolator(coeffs=self.average_coeffs)(points)

    def _average_cdf_from_samples(self, points: np.ndarray) -> np.ndarray:
        if self._average_cdf_interpolator is None:
            _average_cdf = self._create_average_cdf_interpolator()
            object.__setattr__(self, "_average_cdf_interpolator", _average_cdf)

        return self._average_cdf_interpolator(points)

    def _average_pdf_from_polynomials(self, points: np.ndarray) -> np.ndarray:
        return self._variate_pdf_from_polynomials(points, coeffs=self.average_coeffs)

    def _average_pdf_from_samples(self, points: np.ndarray) -> np.ndarray:
        if self._average_pdf_interpolator is None:
            _average_pdf = self._create_average_pdf_interpolator_from_samples()
            object.__setattr__(self, "_average_pdf_interpolator", _average_pdf)

        return self._average_pdf_interpolator(points)

    def _compute_average_coeffs(self) -> np.ndarray:
        average_coeffs = np.zeros(self.max_polynomial_degree + 1)
        for sample in self.sample_stream(realizs=self.optimal_realizs):
            average_coeffs += self.compute_variate_coeffs(sample)

        average_coeffs /= self.optimal_realizs
        return average_coeffs

    def _create_weight_cdf_interpolator(self) -> PchipInterpolator:
        return create_cdf_interpolator_from_pdf(
            self.weight_pdf,
            inputs=np.linspace(*self.plot_range, self.num_pts),
        )

    def _create_average_cdf_interpolator(self) -> PchipInterpolator:
        return create_cdf_interpolator_from_pdf(
            self.average_pdf,
            inputs=np.linspace(*self.plot_range, self.num_pts),
        )

    def _create_average_pdf_interpolator_from_samples(self) -> PchipInterpolator:
        bins = np.linspace(*self.plot_range, self.num_bins + 1)
        counts = np.zeros(self.num_bins)

        for sample in self.sample_stream(self.optimal_realizs):
            counts += np.histogram(sample, bins=bins)[0]

        return create_pdf_interpolator_from_histogram(
            histogram=compute_histogram(counts, bins),
            bins=bins,
            kernel_std_dev=self.kernel_std_dev,
        )

    def _create_variate_pdf_interpolator_from_sample(
        self, sample: np.ndarray
    ) -> PchipInterpolator:
        bins = np.linspace(*self.plot_range, self.num_bins + 1)
        counts = np.histogram(np.asarray(sample), bins=bins)[0]

        return create_pdf_interpolator_from_histogram(
            histogram=compute_histogram(counts, bins),
            bins=bins,
            kernel_std_dev=self.kernel_std_dev,
        )

    def _variate_pdf_from_polynomials(
        self,
        points: np.ndarray,
        *,
        coeffs: np.ndarray | None = None,
        sample: np.ndarray | None = None,
    ) -> np.ndarray:
        if (coeffs is None) == (sample is None):
            raise ValueError("Exactly one of `coeffs` or `sample` must be provided.")

        if sample is not None:
            coeffs = self.compute_variate_coeffs(sample)

        weight_function = self.compute_polynomial_weight(points)
        polynomials = self.compute_polynomials(points)

        return weight_function * np.sum(coeffs[:, None] * polynomials, axis=0)

    def _variate_pdf_from_sample(
        self, points: np.ndarray, *, sample: np.ndarray | None = None
    ) -> np.ndarray:
        if sample is None:
            raise ValueError("`sample` must be provided for sample-based PDFs.")

        pdf_interpolator = self._create_variate_pdf_interpolator_from_sample(sample)
        return pdf_interpolator(points)
