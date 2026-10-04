import numpy as np

from .histograms import (
    COEFFICIENT_GRID_POLICY as COEFFICIENT_GRID_POLICY,
)
from .histograms import (
    CoefficientsHistogram as CoefficientsHistogram,
)

POLYNOMIAL_DEGREE_MIN: int = 1
POLYNOMIAL_DEGREE_STEP: int = 1

REALIZATIONS_METADATA: dict[str, str] = {
    "dir_name": "realizs",
    "latex_name": "R",
}

LOG_D_TIME_SUPPORT: tuple[float, float] = (-0.5, 1.5)

LOG_D_UNFOLDED_TIME_SUPPORT: tuple[float, float] = (-1.5, 0.5)


def nearest_neighbor_spacings(
    values: np.ndarray[tuple[int], np.dtype[np.floating]],
    /,
    *,
    degeneracy: int = 1,
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    spacings = np.diff(np.sort(values))
    if degeneracy > 1:
        spacings = np.repeat(spacings[1::degeneracy], degeneracy)

    return spacings


def scale_support(
    support: tuple[float, float],
    /,
    *,
    scale: float,
) -> tuple[float, float]:
    return scale * support[0], scale * support[1]


def truncated_polynomial_degree_range(*, max_degree: int) -> range:
    return range(POLYNOMIAL_DEGREE_MIN, max_degree + 1, POLYNOMIAL_DEGREE_STEP)
