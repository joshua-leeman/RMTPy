from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def q_hermite_polynomials(
    q: float, max_order: int, x: np.ndarray, out: np.ndarray
) -> np.ndarray:
    out[0, :] = 1.0

    if max_order >= 1:
        out[1, :] = x

    for degree in range(2, max_order + 1):
        if abs(q - 1.0) < 1e-12:
            coeff: float = float(degree)
        else:
            coeff: float = (1.0 - q**degree) / (1.0 - q)

        out[degree, :] = x * out[degree - 1] - coeff * out[degree - 2]

    return out


@njit(cache=True, fastmath=True)
def chebyshev_polynomials_2(
    max_order: int, x: np.ndarray, out: np.ndarray
) -> np.ndarray:

    out[0, :] = 1.0

    if max_order >= 1:
        out[1, :] = 2.0 * x

    for degree in range(2, max_order + 1):
        out[degree, :] = 2.0 * x * out[degree - 1] - out[degree - 2]

    return out
