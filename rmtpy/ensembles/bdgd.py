from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from attrs import field, frozen
from numba import njit

from ._wigner_dyson import WignerDysonEnsemble


@njit(cache=True, fastmath=True)
def _create_bdgd_matrix(
    matrix: np.ndarray,
    rng: np.random.Generator,
    real_dtype: type[np.floating],
    std_dev: float,
) -> np.ndarray:
    size: int = matrix.shape[0]
    for i in range(size):
        matrix[i, i] = 0.0
        matrix[i, i + 1 :] = std_dev * (
            1j * rng.standard_normal(size - i - 1, real_dtype)
        )
        matrix[i + 1 :, i] = np.conj(matrix[i, i + 1 :])


@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class BogoliubovDeGennesDEnsemble(WignerDysonEnsemble):
    dyson_index: int = field(init=False, default=2, repr=False)
    std_dev: float = field(init=False, repr=False)

    @std_dev.default
    def _std_dev_default(self) -> float:
        return self.ground_state_energy / 2 / np.sqrt(self.dimension)

    _nickname: str = field(init=False, default="BdGD", repr=False)

    @property
    def _dir_name(self) -> str:
        return "BdG_D"

    @property
    def _latex_name(self) -> str:
        return "\\textrm{{BdG(D)}}"

    def generate_matrix(self, use_complex_dtype: bool = True) -> np.ndarray:
        complex_dtype: type[np.complexfloating] = self.complex_dtype.type
        real_dtype: type[np.floating] = self.real_dtype.type
        rng: np.random.Generator = self.rng
        size: int = self.dimension
        std_dev: float = self.std_dev

        matrix: np.ndarray = np.empty((size, size), complex_dtype, order="F")
        _create_bdgd_matrix(matrix, rng, real_dtype, std_dev)
        return matrix

    def matrix_stream(
        self, realizs: int, use_complex_dtype: bool = True
    ) -> Iterator[np.ndarray]:
        complex_dtype: type[np.complexfloating] = self.complex_dtype.type
        real_dtype: type[np.floating] = self.real_dtype.type
        rng: np.random.Generator = self.rng
        size: int = self.dimension
        std_dev: float = self.std_dev

        matrix: np.ndarray = np.empty((size, size), complex_dtype, order="F")
        for _ in range(realizs):
            _create_bdgd_matrix(matrix, rng, real_dtype, std_dev)
            yield matrix
