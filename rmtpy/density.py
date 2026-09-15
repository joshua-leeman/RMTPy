from collections.abc import Callable, Iterator
from functools import cached_property
from typing import Protocol, cast

import attrs
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

import rmtpy.validators
from rmtpy.polynomials import Float64Function, OrthogonalPolynomials


class SpectrumStream(Protocol):
    def __call__(
        self,
        realizs: int,
        *,
        use_complex_dtype: bool = False,
    ) -> Iterator[NDArray[np.floating]]: ...


MAX_POLYNOMIAL_DEGREE: int = 6

SUPPORT_SCALE_FACTOR: float = 1.2

NUM_REALIZATIONS_MIN: int = 10

NUM_HISTOGRAM_COUNTS: int = 2**13

GAUSSIAN_KERNEL_STANDARD_DEVIATION: float = 2.0

NUM_POINTS: int = 1000


def array_of_floats(
    *,
    support: tuple[float, float],
    num_pts: int,
    log_base: float | None = None,
) -> NDArray[np.float64]:
    rmtpy.validators.validate_support(support)
    if log_base is None:
        return np.linspace(*support, num_pts)
    else:
        return np.logspace(*support, num_pts, base=log_base)


def compute_bin_centers(bins: NDArray[np.float64]) -> NDArray[np.float64]:
    if bins.ndim != 1:
        raise ValueError("`bins` must be one-dimensional.")
    if len(bins) < 2:
        raise ValueError("At least two histogram bin edges are required.")
    if np.any(np.diff(bins) <= 0):
        raise ValueError("`bins` must be strictly increasing.")

    neighbor_ratio = cast(np.float64, bins[1] / bins[0])
    if np.all(bins > 0.0) and np.allclose(bins[1:] / bins[:-1], neighbor_ratio):
        return np.sqrt(bins[:-1] * bins[1:])

    return (bins[:-1] + bins[1:]) / 2


def compute_histogram(
    counts: NDArray[np.intp],
    *,
    bins: NDArray[np.float64],
) -> NDArray[np.float64]:
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
    histogram: NDArray[np.float64],
    *,
    bins: NDArray[np.float64],
    kernel_std_dev: float = GAUSSIAN_KERNEL_STANDARD_DEVIATION,
) -> PchipInterpolator:
    centers = compute_bin_centers(bins)
    pdf_values = gaussian_filter1d(histogram, kernel_std_dev)

    return PchipInterpolator(centers, pdf_values, extrapolate=True)


def create_cdf_interpolator_from_pdf(
    pdf: Float64Function,
    *,
    inputs: NDArray[np.float64],
    left_tail_mass: float = 0.0,
) -> PchipInterpolator:
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

    integral = cumulative_trapezoid(pdf(inputs), inputs, initial=0)
    cdf_values = left_tail_mass + integral

    return PchipInterpolator(inputs, cdf_values, extrapolate=True)


def unfold_values_with_cdf(
    values: NDArray[np.float64],
    *,
    cdf: Float64Function | None = None,
    dimension: int,
) -> NDArray[np.float64]:
    if cdf is None:
        return values

    return dimension * (cdf(values) - cdf(np.array([0.0])))


def unfold_widths_with_cdf(
    *,
    widths: NDArray[np.float64],
    centers: NDArray[np.float64],
    cdf: Float64Function | None = None,
    dimension: int,
) -> NDArray[np.float64]:
    if cdf is None:
        return widths

    return dimension * (cdf(centers + widths / 2) - cdf(centers - widths / 2))


def _compute_default_number_of_bins(density: DensityModel) -> int:
    rough_estimate = cast(np.float64, np.sqrt(density.dimension))
    rounded_estimate = cast(np.float64, np.ceil(rough_estimate))
    return max(int(rounded_estimate), 2)


def _compute_optimal_realizations(density: DensityModel) -> int:
    return max(NUM_HISTOGRAM_COUNTS // density.dimension, NUM_REALIZATIONS_MIN)


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class DensityModel:
    max_polynomial_degree: int = attrs.field(
        default=MAX_POLYNOMIAL_DEGREE,
        converter=int,
        validator=attrs.validators.ge(0),
    )
    polynomials: OrthogonalPolynomials | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(attrs.validators.is_callable()),
    )
    weight_function: Float64Function | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(attrs.validators.is_callable()),
    )
    sample_stream: SpectrumStream = attrs.field(
        validator=attrs.validators.is_callable(),
        repr=False,
    )
    dimension: int = attrs.field(
        converter=int,
        validator=attrs.validators.gt(0),
    )
    support: tuple[float, float] = attrs.field(
        converter=cast(Callable[[object], tuple[float, float]], tuple),
        validator=rmtpy.validators.is_valid_support,
    )

    support_scale_factor: float = attrs.field(
        default=SUPPORT_SCALE_FACTOR,
        converter=float,
        validator=attrs.validators.ge(1.0),
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
        validator=attrs.validators.gt(1),
        repr=False,
    )
    num_bins: int = attrs.field(
        default=attrs.Factory(_compute_default_number_of_bins, takes_self=True),
        converter=int,
        validator=attrs.validators.gt(1),
        repr=False,
    )
    optimal_realizs: int = attrs.field(
        default=attrs.Factory(_compute_optimal_realizations, takes_self=True),
        init=False,
        repr=False,
    )

    @property
    def has_polynomial_expansion(self) -> bool:
        return self.polynomials is not None and self.weight_function is not None

    @property
    def support_radius(self) -> float:
        return (self.support[1] - self.support[0]) / 2

    @property
    def plot_range(self) -> tuple[float, float]:
        center = sum(self.support) / 2
        radius = self.support_scale_factor * self.support_radius
        return center - radius, center + radius

    @cached_property
    def average_coeffs(self) -> NDArray[np.float64]:
        return self._compute_average_coeffs()

    @cached_property
    def _average_pdf_interpolator(self) -> PchipInterpolator:
        return self._create_average_pdf_interpolator_from_samples()

    @cached_property
    def _average_cdf_interpolator(self) -> PchipInterpolator:
        return self._create_average_cdf_interpolator()

    @cached_property
    def _weight_cdf_interpolator(self) -> PchipInterpolator:
        return self._create_weight_cdf_interpolator()

    def average_cdf(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self.has_polynomial_expansion:
            return self._average_cdf_from_samples(points)

        return self._average_cdf_from_polynomials(points)

    def average_pdf(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self.has_polynomial_expansion:
            return self._average_pdf_from_samples(points)

        return self._average_pdf_from_polynomials(points)

    def compute_polynomials(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.polynomials is None:
            raise NotImplementedError()

        center = sum(self.support) / 2
        x = (inputs - center) / self.support_radius

        return self.polynomials(x, degree=self.max_polynomial_degree)

    def compute_polynomial_weight(
        self,
        inputs: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if self.weight_function is None:
            raise NotImplementedError()

        return self.weight_function(inputs)

    def compute_variate_coeffs(
        self,
        sample: NDArray[np.floating],
    ) -> NDArray[np.float64]:
        polynomials = self.compute_polynomials(sample.astype(np.float64))
        return cast(NDArray[np.float64], np.mean(polynomials, axis=1))

    def create_variate_cdf_interpolator(
        self,
        *,
        interval: tuple[float, float] | None = None,
        coeffs: NDArray[np.float64] | None = None,
        sample: NDArray[np.float64] | None = None,
    ) -> PchipInterpolator:
        if interval is None:
            interval = self.plot_range
        else:
            rmtpy.validators.validate_support(interval)

        def pdf(points: NDArray[np.float64]) -> NDArray[np.float64]:
            return self.variate_pdf(points, coeffs=coeffs, sample=sample)

        if interval[0] > self.plot_range[0]:
            left_support = (self.plot_range[0], interval[0])
            left_tail = array_of_floats(support=left_support, num_pts=self.num_pts)
            left_mass = cast(float, cumulative_trapezoid(pdf(left_tail), left_tail)[-1])
        else:
            left_mass = 0.0

        return create_cdf_interpolator_from_pdf(
            pdf,
            inputs=array_of_floats(support=interval, num_pts=self.num_pts),
            left_tail_mass=left_mass,
        )

    def variate_pdf(
        self,
        points: NDArray[np.float64],
        *,
        coeffs: NDArray[np.float64] | None = None,
        sample: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        if self.has_polynomial_expansion:
            return self._variate_pdf_from_polynomials(
                points, coeffs=coeffs, sample=sample
            )

        return self._variate_pdf_from_sample(points, sample=sample)

    def variate_cdf(
        self,
        points: NDArray[np.float64],
        *,
        coeffs: NDArray[np.float64] | None = None,
        sample: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        cdf_interpolator = self.create_variate_cdf_interpolator(
            coeffs=coeffs, sample=sample
        )
        return cdf_interpolator(points)

    def weight_pdf(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self.has_polynomial_expansion:
            return self._average_pdf_from_samples(points)

        return self.compute_polynomial_weight(points)

    def weight_cdf(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self.has_polynomial_expansion:
            return self._average_cdf_from_samples(points)

        return self._weight_cdf_interpolator(points)

    def _average_cdf_from_polynomials(
        self, points: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return self.create_variate_cdf_interpolator(coeffs=self.average_coeffs)(points)

    def _average_cdf_from_samples(
        self, points: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return self._average_cdf_interpolator(points)

    def _average_pdf_from_polynomials(
        self, points: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return self._variate_pdf_from_polynomials(points, coeffs=self.average_coeffs)

    def _average_pdf_from_samples(
        self, points: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return self._average_pdf_interpolator(points)

    def _compute_average_coeffs(self) -> NDArray[np.float64]:
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
        counts = np.zeros(self.num_bins, dtype=np.intp)

        for sample in self.sample_stream(self.optimal_realizs):
            counts += np.histogram(sample, bins=bins)[0]

        return create_pdf_interpolator_from_histogram(
            histogram=compute_histogram(counts, bins=bins),
            bins=bins,
            kernel_std_dev=self.kernel_std_dev,
        )

    def _create_variate_pdf_interpolator_from_sample(
        self,
        sample: NDArray[np.float64],
    ) -> PchipInterpolator:
        bins = np.linspace(*self.plot_range, self.num_bins + 1)
        counts = np.histogram(sample, bins=bins)[0]

        return create_pdf_interpolator_from_histogram(
            histogram=compute_histogram(counts, bins=bins),
            bins=bins,
            kernel_std_dev=self.kernel_std_dev,
        )

    def _variate_pdf_from_polynomials(
        self,
        points: NDArray[np.float64],
        *,
        coeffs: NDArray[np.float64] | None = None,
        sample: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        if (coeffs is None) == (sample is None):
            raise ValueError("Exactly one of `coeffs` or `sample` must be provided.")

        if sample is not None:
            expansion_coeffs = self.compute_variate_coeffs(sample)
        else:
            expansion_coeffs = cast(NDArray[np.float64], coeffs)

        weight_function = self.compute_polynomial_weight(points)
        polynomials = self.compute_polynomials(points)

        expansion_factor = cast(
            NDArray[np.float64], np.sum(expansion_coeffs[:, None] * polynomials, axis=0)
        )
        return weight_function * expansion_factor

    def _variate_pdf_from_sample(
        self,
        points: NDArray[np.float64],
        *,
        sample: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        if sample is None:
            raise ValueError("`sample` must be provided for sample-based PDFs.")

        pdf_interpolator = self._create_variate_pdf_interpolator_from_sample(sample)
        return pdf_interpolator(points)
