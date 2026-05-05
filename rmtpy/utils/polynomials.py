from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def q_hermite_polynomials(q: float, max_degree: int, x: np.ndarray) -> np.ndarray:
    polynomials: np.ndarray = np.empty((max_degree + 1, x.size))

    polynomials[0, :] = 1.0

    if max_degree >= 1:
        polynomials[1, :] = x

    for degree in range(2, max_degree + 1):
        if abs(q - 1.0) < 1e-12:
            coeff: float = float(degree)
        else:
            coeff: float = (1.0 - q**degree) / (1.0 - q)

        polynomials[degree, :] = (
            x * polynomials[degree - 1] - coeff * polynomials[degree - 2]
        )

    return polynomials


@njit(cache=True, fastmath=True)
def chebyshev_polynomials_2(max_degree: int, x: np.ndarray) -> np.ndarray:
    polynomials: np.ndarray = np.empty((max_degree + 1, x.size))

    polynomials[0, :] = 1.0

    if max_degree >= 1:
        polynomials[1, :] = 2.0 * x

    for degree in range(2, max_degree + 1):
        polynomials[degree, :] = (
            2.0 * x * polynomials[degree - 1] - polynomials[degree - 2]
        )

    return polynomials
