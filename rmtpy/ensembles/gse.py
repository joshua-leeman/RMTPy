from __future__ import annotations

from collections.abc import Iterator
from typing import ClassVar

import attrs
import numba
import numpy as np

from .gue import create_gue_matrix
from .wigner_dyson import WignerDysonEnsemble

INITIALISM: str = "GSE"

DYSON_INDEX: int = 4


def compute_standard_deviation(gse: GaussianSymplecticEnsemble) -> float:
    return gse.spectral_radius / 2 / np.sqrt(2 * gse.dimension)


@numba.njit(cache=True, fastmath=True)
def create_skew_matrix(
    matrix: np.ndarray,
    rng: np.random.Generator,
    real_dtype: type[np.floating],
    std_dev: float,
) -> np.ndarray:
    size: int = matrix.shape[0]
    for i in range(size):
        matrix[i, i] = 0.0
        matrix[i + 1 :, i] = std_dev * (
            rng.standard_normal(size - 1 - i, real_dtype)
            + 1j * rng.standard_normal(size - 1 - i, real_dtype)
        )
        matrix[i, i + 1 :] = -matrix[i + 1 :, i]


def create_gse_matrix(
    matrix: np.ndarray,
    rng: np.random.Generator,
    real_dtype: type[np.floating],
    std_dev: float,
) -> np.ndarray:
    halfway: int = matrix.shape[0] // 2
    top_left_block = matrix[:halfway, :halfway]
    top_right_block = matrix[:halfway, halfway:]
    bottom_left_block = matrix[halfway:, :halfway]
    bottom_right_block = matrix[halfway:, halfway:]

    create_gue_matrix(top_left_block, rng, real_dtype, std_dev)
    np.conj(top_left_block, out=bottom_right_block)

    create_skew_matrix(top_right_block, rng, real_dtype, std_dev)
    np.negative(top_right_block, out=bottom_left_block)
    np.conj(bottom_left_block, out=bottom_left_block)


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class GaussianSymplecticEnsemble(WignerDysonEnsemble):
    initialism: ClassVar[str] = INITIALISM

    std_dev: float = attrs.field(
        default=attrs.Factory(compute_standard_deviation, takes_self=True),
        init=False,
        repr=False,
    )
    dyson_index: int = attrs.field(
        default=DYSON_INDEX,
        init=False,
        repr=False,
    )

    def generate_matrix(self, use_complex_dtype: bool = False) -> np.ndarray:
        matrix: np.ndarray = self._initialize_matrix(use_complex_dtype)
        create_gse_matrix(matrix, self.rng, self.real_dtype.type, self.std_dev)
        return matrix

    def matrix_stream(
        self, realizs: int, use_complex_dtype: bool = False
    ) -> Iterator[np.ndarray]:
        matrix: np.ndarray = self._initialize_matrix(use_complex_dtype)
        for _ in range(realizs):
            create_gse_matrix(matrix, self.rng, self.real_dtype.type, self.std_dev)
            yield matrix
