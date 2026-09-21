from collections.abc import Iterable
from typing import cast

import attrs
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator

from ..density import (
    DensityModel,
    array_of_floats,
    unfold_values_with_cdf,
    unfold_widths_with_cdf,
)
from ..polynomials import Float64Function


def normalize_degrees(degrees: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted(int(degree) for degree in degrees))


def unfold_values(
    values: NDArray[np.float64],
    *,
    cdf: Float64Function,
    dimension: int,
) -> NDArray[np.float64]:
    return unfold_values_with_cdf(values, cdf=cdf, dimension=dimension)


def unfold_widths(
    widths: NDArray[np.float64],
    centers: NDArray[np.float64],
    *,
    cdf: Float64Function,
    dimension: int,
) -> NDArray[np.float64]:
    return unfold_widths_with_cdf(
        widths=widths,
        centers=centers,
        cdf=cdf,
        dimension=dimension,
    )


def _build_grid(factory: TruncatedPolynomialCdfFactory) -> NDArray[np.float64]:
    if not factory.degrees:
        return np.empty(0, dtype=np.float64)

    return array_of_floats(
        support=factory.density.plot_range,
        num_pts=factory.density.num_pts,
    )


def _compute_polynomials(factory: TruncatedPolynomialCdfFactory) -> NDArray[np.float64]:
    if not factory.degrees:
        return np.empty((0, 0), dtype=np.float64)

    return factory.density.compute_polynomials(factory.grid)


def _compute_weight(factory: TruncatedPolynomialCdfFactory) -> NDArray[np.float64]:
    if not factory.degrees:
        return np.empty(0, dtype=np.float64)

    return factory.density.compute_polynomial_weight(factory.grid)


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class TruncatedPolynomialCdfFactory:
    density: DensityModel = attrs.field(
        validator=attrs.validators.instance_of(DensityModel),
    )
    density_name: str = attrs.field(
        default="spectral",
        converter=str,
        validator=attrs.validators.min_len(1),
    )
    degrees: tuple[int, ...] = attrs.field(
        converter=normalize_degrees,
        validator=attrs.validators.deep_iterable(
            member_validator=(
                attrs.validators.instance_of(int),
                attrs.validators.ge(0),
            ),
        ),
        repr=False,
    )

    grid: NDArray[np.float64] = attrs.field(
        default=attrs.Factory(_build_grid, takes_self=True),
        init=False,
        repr=False,
    )
    polynomials: NDArray[np.float64] = attrs.field(
        default=attrs.Factory(_compute_polynomials, takes_self=True),
        init=False,
        repr=False,
    )
    weight: NDArray[np.float64] = attrs.field(
        default=attrs.Factory(_compute_weight, takes_self=True),
        init=False,
        repr=False,
    )

    def __attrs_post_init__(self) -> None:
        if not self.degrees:
            return

        if not self.density.has_polynomial_expansion:
            raise NotImplementedError(
                "Truncated polynomial unfolding requires a polynomial "
                + f"{self.density_name} density expansion."
            )

        max_degree = max(self.degrees)
        if max_degree > self.density.max_polynomial_degree:
            raise ValueError("Truncated CDF degree cannot exceed DensityModel degree.")

    def average_interpolators(self) -> tuple[PchipInterpolator, ...]:
        if not self.degrees:
            return ()

        coeffs = self.density.average_coeffs
        return self.interpolators_from_coeffs(coeffs)

    def build_interpolator(self, pdf_values: NDArray[np.float64]) -> PchipInterpolator:
        cdf_values = cumulative_trapezoid(y=pdf_values, x=self.grid, initial=0)
        return PchipInterpolator(x=self.grid, y=cdf_values, extrapolate=True)

    def interpolators_from_coeffs(
        self,
        coeffs: NDArray[np.float64],
    ) -> tuple[PchipInterpolator, ...]:
        coeffs = np.asarray(coeffs)
        if self.degrees and len(coeffs) <= max(self.degrees):
            raise ValueError("Coefficient array is shorter than the requested degree.")

        interpolators: list[PchipInterpolator] = []
        polynomial_sum = np.zeros_like(self.grid, dtype=np.float64)
        next_degree = 0

        for degree in self.degrees:
            polynomial_sum += cast(
                NDArray[np.floating],
                np.sum(
                    coeffs[next_degree : degree + 1, None]
                    * self.polynomials[next_degree : degree + 1],
                    axis=0,
                ),
            )
            next_degree = degree + 1
            interpolators.append(self.build_interpolator(self.weight * polynomial_sum))

        return tuple(interpolators)
