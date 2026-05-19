from __future__ import annotations

from collections.abc import Iterator
from typing import ClassVar

import attrs
import numba
import numpy as np

from .wigner_dyson import WignerDysonEnsemble

INITIALISM: str = "GUE"

DYSON_INDEX: int = 2


def compute_standard_deviation(gue: GaussianUnitaryEnsemble) -> float:
    return gue.spectral_radius / 2 / np.sqrt(2 * gue.dimension)


@numba.njit(cache=True, fastmath=True)
def create_gue_matrix(
    matrix: np.ndarray,
    rng: np.random.Generator,
    real_dtype: type[np.floating],
    std_dev: float,
) -> np.ndarray:
    size: int = matrix.shape[0]
    for i in range(size):
        matrix[i, i] = 2 * std_dev * rng.standard_normal(None, real_dtype)
        matrix[i + 1 :, i] = std_dev * (
            rng.standard_normal(size - i - 1, real_dtype)
            + 1j * rng.standard_normal(size - i - 1, real_dtype)
        )
        matrix[i, i + 1 :] = np.conj(matrix[i + 1 :, i])


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class GaussianUnitaryEnsemble(WignerDysonEnsemble):
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
        create_gue_matrix(matrix, self.rng, self.real_dtype.type, self.std_dev)
        return matrix

    def matrix_stream(
        self, realizs: int, use_complex_dtype: bool = False
    ) -> Iterator[np.ndarray]:
        matrix: np.ndarray = self._initialize_matrix(use_complex_dtype)
        for _ in range(realizs):
            create_gue_matrix(matrix, self.rng, self.real_dtype.type, self.std_dev)
            yield matrix
