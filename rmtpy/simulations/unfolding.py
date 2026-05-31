from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeAlias

import attrs
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator

import rmtpy.density

CDF: TypeAlias = Callable[[np.ndarray], np.ndarray]


def normalize_degrees(degrees: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted(int(degree) for degree in degrees))


def unfold_time_delays(
    time_delays: np.ndarray,
    *,
    energy: float,
    cdf: CDF,
    dimension: int,
) -> np.ndarray:
    valid_delays = time_delays[np.isfinite(time_delays) & (time_delays > 0.0)]
    if valid_delays.size == 0:
        return valid_delays

    widths = np.reciprocal(valid_delays)
    unfolded_widths = unfold_widths(
        widths,
        np.full_like(widths, energy),
        cdf=cdf,
        dimension=dimension,
    )
    valid_unfolded_widths = unfolded_widths[
        np.isfinite(unfolded_widths) & (unfolded_widths > 0.0)
    ]
    return np.reciprocal(valid_unfolded_widths)


def unfold_values(
    values: np.ndarray,
    *,
    cdf: CDF,
    dimension: int,
) -> np.ndarray:
    return rmtpy.density.unfold_values_with_cdf(
        values,
        cdf=cdf,
        dimension=dimension,
    )


def unfold_widths(
    widths: np.ndarray,
    centers: np.ndarray,
    *,
    cdf: CDF,
    dimension: int,
) -> np.ndarray:
    return rmtpy.density.unfold_widths_with_cdf(
        widths=widths,
        centers=centers,
        cdf=cdf,
        dimension=dimension,
    )


def _compute_polynomials(factory: TruncatedPolynomialCdfFactory) -> np.ndarray:
    return factory.density.compute_polynomials(factory.grid)


def _compute_weight(factory: TruncatedPolynomialCdfFactory) -> np.ndarray:
    return factory.density.compute_polynomial_weight(factory.grid)


def _create_grid(factory: TruncatedPolynomialCdfFactory) -> np.ndarray:
    return rmtpy.density.array_of_floats(
        support=factory.density.plot_range,
        num_pts=factory.density.num_pts,
    )


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class TruncatedPolynomialCdfFactory:
    density: rmtpy.density.DensityModel = attrs.field(
        validator=attrs.validators.instance_of(rmtpy.density.DensityModel),
    )
    density_name: str = attrs.field(
        default="spectral",
        converter=str,
        validator=attrs.validators.min_len(1),
    )
    degrees: tuple[int, ...] = attrs.field(
        converter=normalize_degrees,
        validator=attrs.validators.deep_iterable(
            member_validator=[
                attrs.validators.instance_of(int),
                attrs.validators.ge(0),
            ],
            iterable_validator=attrs.validators.min_len(1),
        ),
        repr=False,
    )

    grid: np.ndarray = attrs.field(
        default=attrs.Factory(_create_grid, takes_self=True),
        init=False,
        repr=False,
    )
    polynomials: np.ndarray = attrs.field(
        default=attrs.Factory(_compute_polynomials, takes_self=True),
        init=False,
        repr=False,
    )
    weight: np.ndarray = attrs.field(
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
                f"{self.density_name} density expansion."
            )

        max_degree = max(self.degrees)
        if max_degree > self.density.max_polynomial_degree:
            raise ValueError(
                "Truncated CDF degree cannot exceed the density polynomial degree."
            )

    def average_interpolators(self) -> tuple[PchipInterpolator, ...]:
        if not self.degrees:
            return ()

        coeffs = self.density.average_coeffs
        if coeffs is None:
            raise NotImplementedError(
                "Truncated average unfolding requires average polynomial coefficients."
            )

        return self.interpolators_from_coeffs(coeffs)

    def interpolators_from_coeffs(
        self,
        coeffs: np.ndarray,
    ) -> tuple[PchipInterpolator, ...]:
        coeffs = np.asarray(coeffs)
        if self.degrees and len(coeffs) <= max(self.degrees):
            raise ValueError("Coefficient array is shorter than the requested degree.")

        interpolators: list[PchipInterpolator] = []
        polynomial_sum = np.zeros_like(self.grid, dtype=np.float64)
        next_degree = 0

        for degree in self.degrees:
            polynomial_sum += np.sum(
                coeffs[next_degree : degree + 1, None]
                * self.polynomials[next_degree : degree + 1],
                axis=0,
            )
            next_degree = degree + 1
            interpolators.append(
                self._create_interpolator(self.weight * polynomial_sum)
            )

        return tuple(interpolators)

    def _create_interpolator(self, pdf_values: np.ndarray) -> PchipInterpolator:
        cdf_values = cumulative_trapezoid(pdf_values, self.grid, initial=0.0)
        return PchipInterpolator(self.grid, cdf_values, extrapolate=True)
