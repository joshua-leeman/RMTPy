from collections.abc import Callable
from typing import Protocol, cast

import numba
import numpy as np
from numpy.typing import NDArray

type RealFunction = Callable[[NDArray[np.floating]], NDArray[np.floating]]


class OrthogonalPolynomials(Protocol):
    def __call__(
        self,
        x: NDArray[np.floating],
        *,
        degree: int,
    ) -> NDArray[np.floating]: ...


def chebyshev_polynomial_2_weight(
    energies: NDArray[np.float64],
    *,
    support_radius: float,
) -> NDArray[np.float64]:
    energies = np.asarray(energies)
    x = energies / support_radius

    in_support = np.abs(energies) < support_radius
    pdf = np.zeros_like(x, dtype=np.result_type(x, np.float64))
    pdf[in_support] = 2 / np.pi / support_radius * np.sqrt(1.0 - x[in_support] ** 2)

    return pdf


@numba.njit(boundscheck=False, cache=True, fastmath=True)
def chebyshev_polynomials_2(
    x: NDArray[np.float64],
    *,
    degree: int,
) -> NDArray[np.float64]:
    polynomials = np.empty((degree + 1, x.size), dtype=np.float64)

    polynomials[0, :] = 1.0
    if degree >= 1:
        polynomials[1, :] = 2.0 * x

    for n in range(2, degree + 1):
        polynomials[n, :] = 2.0 * x * polynomials[n - 1] - polynomials[n - 2]

    return polynomials


def legendre_polynomial_weight(
    energies: NDArray[np.float64],
    *,
    support_radius: float,
) -> NDArray[np.float64]:
    energies = np.asarray(energies)
    x = energies / support_radius

    in_support = np.abs(energies) < support_radius
    pdf = np.zeros_like(x, dtype=np.result_type(x, np.float64))
    pdf[in_support] = 1 / (2 * support_radius)

    return pdf


@numba.njit(boundscheck=False, cache=True, fastmath=True)
def legendre_polynomials(
    x: NDArray[np.float64],
    *,
    degree: int,
) -> NDArray[np.float64]:
    polynomials = np.empty((degree + 1, x.size), dtype=np.float64)

    polynomials[0, :] = 1.0

    if degree >= 1:
        polynomials[1, :] = x

    for n in range(2, degree + 1):
        polynomials[n, :] = (
            (2 * n - 1) * x * polynomials[n - 1] - (n - 1) * polynomials[n - 2]
        ) / n

    return polynomials


def q_hermite_polynomial_weight(
    energies: NDArray[np.float64],
    *,
    support_radius: float,
    eta: float,
    partial_product_order: int = 100,
) -> NDArray[np.float64]:
    energies = np.asarray(energies)
    x = energies / support_radius

    index_range = np.arange(partial_product_order)
    etak1 = eta ** (index_range + 1)

    in_support = np.abs(energies) < support_radius
    prefactor = np.zeros_like(x, dtype=np.result_type(x, np.float64))
    term1 = 1.0 - (4 * x[in_support][:, None] ** 2) * etak1 / (1.0 + etak1) ** 2
    term2 = (1.0 - eta ** (2 * index_range + 2)) / (1.0 - eta ** (2 * index_range + 1))

    log_terms = np.log(term1) + np.log(term2)[None, :]
    prefactor[in_support] = np.exp(cast(NDArray[np.float64], np.sum(log_terms, axis=1)))

    return prefactor * chebyshev_polynomial_2_weight(
        energies, support_radius=support_radius
    )


@numba.njit(boundscheck=False, cache=True, fastmath=True)
def q_hermite_polynomials(
    x: NDArray[np.float64],
    *,
    eta: float,
    degree: int,
) -> NDArray[np.float64]:
    polynomials = np.empty((degree + 1, x.size), dtype=np.float64)

    polynomials[0, :] = 1.0

    if degree >= 1:
        polynomials[1, :] = 2 / np.sqrt(1 - eta) * x

    for n in range(2, degree + 1):
        if abs(eta - 1.0) < 1e-12:
            eta_num = float(n)
        else:
            eta_num = (1.0 - eta ** (n - 1)) / (1.0 - eta)

        polynomials[n, :] = (
            2 / np.sqrt(1 - eta) * x * polynomials[n - 1] - eta_num * polynomials[n - 2]
        )

    norms = np.empty(degree + 1, dtype=np.float64)
    norms[0] = 1.0

    norm_squared = 1.0
    for k in range(1, degree + 1):
        if abs(eta - 1.0) < 1e-12:
            norm_squared *= float(k)
        else:
            norm_squared *= (1.0 - eta**k) / (1.0 - eta)

        norms[k] = np.sqrt(norm_squared)

    polynomials /= norms[:, None]

    return polynomials
