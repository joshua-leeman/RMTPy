import numba
import numpy as np


def chebyshev_polynomial_2_weight_pdf(
    energies: np.ndarray, *, radius: float
) -> np.ndarray:
    energies = np.asarray(energies)
    x = energies / radius

    in_support = np.abs(energies) < radius
    pdf = np.zeros_like(x, dtype=np.result_type(x, np.float64))
    pdf[in_support] = 2 / np.pi / radius * np.sqrt(1.0 - x[in_support] ** 2)

    return pdf


@numba.njit(boundscheck=False, cache=True, fastmath=True)
def chebyshev_polynomials_2(x: np.ndarray, *, degree: int) -> np.ndarray:
    polynomials = np.empty((degree + 1, x.size), dtype=np.float64)

    polynomials[0, :] = 1.0
    if degree >= 1:
        polynomials[1, :] = 2.0 * x

    for n in range(2, degree + 1):
        polynomials[n, :] = 2.0 * x * polynomials[n - 1] - polynomials[n - 2]

    return polynomials


def legendre_polynomial_weight_pdf(energies: np.ndarray, *, radius: float) -> np.ndarray:
    energies = np.asarray(energies)
    x = energies / radius

    in_support = np.abs(energies) < radius
    pdf = np.zeros_like(x, dtype=np.result_type(x, np.float64))
    pdf[in_support] = 1 / (2 * radius)

    return pdf


@numba.njit(boundscheck=False, cache=True, fastmath=True)
def legendre_polynomials(x: np.ndarray, *, degree: int) -> np.ndarray:
    polynomials = np.empty((degree + 1, x.size), dtype=np.float64)

    polynomials[0, :] = 1.0

    if degree >= 1:
        polynomials[1, :] = x

    for n in range(2, degree + 1):
        polynomials[n, :] = (
            (2 * n - 1) * x * polynomials[n - 1] - (n - 1) * polynomials[n - 2]
        ) / n

    return polynomials


def q_hermite_polynomial_weight_pdf(
    energies: np.ndarray,
    *,
    radius: float,
    eta: float,
    partial_product_order: int = 100,
) -> np.ndarray:
    energies = np.asarray(energies)
    x = energies / radius

    k = np.arange(partial_product_order)
    etak1 = eta ** (k + 1)

    in_support = np.abs(energies) < radius
    product = np.zeros_like(x, dtype=np.result_type(x, np.float64))
    term1 = 1.0 - (4 * x[in_support][:, None] ** 2) * etak1 / (1.0 + etak1) ** 2
    term2 = (1.0 - eta ** (2 * k + 2)) / (1.0 - eta ** (2 * k + 1))
    product[in_support] = np.exp(np.sum(np.log(term1) + np.log(term2)[None, :], axis=1))

    return product * chebyshev_polynomial_2_weight_pdf(energies, radius=radius)


@numba.njit(boundscheck=False, cache=True, fastmath=True)
def q_hermite_polynomials(x: np.ndarray, *, eta: float, degree: int) -> np.ndarray:
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
