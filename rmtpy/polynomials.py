import numba
import numpy as np


def constant_weight_pdf(energies: np.ndarray, spectral_radius: float) -> np.ndarray:
    in_support: np.ndarray = np.abs(energies) < spectral_radius
    x: np.ndarray = np.asarray(energies)[in_support] / spectral_radius

    pdf: np.ndarray = np.zeros_like(x, dtype=np.result_type(x, np.float64))
    pdf[in_support] = 1 / (2 * spectral_radius)
    return pdf


@numba.njit(cache=True, fastmath=True)
def legendre_polynomials(x: np.ndarray, degree: int) -> np.ndarray:
    polynomials: np.ndarray = np.empty((degree + 1, x.size), dtype=np.float64)

    polynomials[0, :] = 1.0

    if degree >= 1:
        polynomials[1, :] = x

    for n in range(2, degree + 1):
        polynomials[n, :] = (
            ((2 * n - 1) * x * polynomials[n - 1] - (n - 1) * polynomials[n - 2]) / n,
        )


def semicircle_weight_pdf(energies: np.ndarray, spectral_radius: float) -> np.ndarray:
    in_support: np.ndarray = np.abs(energies) < spectral_radius
    x: np.ndarray = np.asarray(energies)[in_support] / spectral_radius

    pdf: np.ndarray = np.zeros_like(x, dtype=np.result_type(x, np.float64))
    pdf[in_support] = 2 / np.pi / spectral_radius * np.sqrt(1.0 - x[in_support] ** 2)
    return pdf


@numba.njit(cache=True, fastmath=True)
def chebyshev_polynomials_2(x: np.ndarray, degree: int) -> np.ndarray:
    polynomials: np.ndarray = np.empty((degree + 1, x.size), dtype=np.float64)

    polynomials[0, :] = 1.0

    if degree >= 1:
        polynomials[1, :] = 2.0 * x

    for n in range(2, degree + 1):
        polynomials[n, :] = 2.0 * x * polynomials[n - 1] - polynomials[n - 2]

    return polynomials


def q_hermite_polynomial_weight_pdf(
    energies: np.ndarray,
    spectral_radius: float,
    eta: float,
    partial_product_order: int = 100,
) -> np.ndarray:
    in_support: np.ndarray = np.abs(energies) < spectral_radius
    x: np.ndarray = np.asarray(energies)[in_support] / spectral_radius

    k: np.ndarray = np.arange(partial_product_order)
    etak1: np.ndarray = eta ** (k + 1)

    term1: np.ndarray = 1.0 - (4 * x[:, None] ** 2) * etak1 / (1.0 + etak1) ** 2
    term2: np.ndarray = (1.0 - eta ** (2 * k + 2)) / (1.0 - eta ** (2 * k + 1))
    product: np.ndarray = np.exp(np.sum(np.log(term1) + np.log(term2)[None, :], axis=1))

    pdf: np.ndarray = np.zeros_like(x, dtype=np.result_type(x, np.float64))
    pdf[in_support] = semicircle_weight_pdf(energies, spectral_radius) * product
    return pdf


@numba.njit(cache=True, fastmath=True)
def q_hermite_polynomials(x: np.ndarray, eta: float, degree: int) -> np.ndarray:
    polynomials: np.ndarray = np.empty((degree + 1, x.size), dtype=np.float64)

    polynomials[0, :] = 1.0

    if degree >= 1:
        polynomials[1, :] = x

    for n in range(2, degree + 1):
        if abs(eta - 1.0) < 1e-12:
            coeff: float = float(n)
        else:
            coeff: float = (1.0 - eta**n) / (1.0 - eta)

        polynomials[n, :] = x * polynomials[n - 1] - coeff * polynomials[n - 2]

    return polynomials
