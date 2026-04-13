from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from attrs import field, frozen
from numba import njit

from ._wigner_dyson import WignerDysonEnsemble
from .gue import _create_gue_matrix


@njit(cache=True, fastmath=True)
def _create_symm_matrix(
    matrix: np.ndarray,
    rng: np.random.Generator,
    real_dtype: type[np.floating],
    std_dev: float,
) -> np.ndarray:
    size: int = matrix.shape[0]
    for i in range(size):
        matrix[i, i] = 2 * std_dev * rng.standard_normal(None, real_dtype)
        matrix[i, i + 1 :] = std_dev * (
            rng.standard_normal(size - 1 - i, real_dtype)
            + 1j * rng.standard_normal(size - 1 - i, real_dtype)
        )
        matrix[i + 1 :, i] = matrix[i, i + 1 :]


def _create_bdgc_matrix(
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

    _create_gue_matrix(top_left_block, rng, real_dtype, std_dev)
    np.negative(top_left_block, out=bottom_right_block)
    np.conj(bottom_right_block, out=bottom_right_block)

    _create_symm_matrix(top_right_block, rng, real_dtype, std_dev)
    np.conj(top_right_block, out=bottom_left_block)


@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class BogoliubovDeGennesCEnsemble(WignerDysonEnsemble):
    dyson_index: int = field(init=False, default=2, repr=False)
    std_dev: float = field(init=False, repr=False)

    @std_dev.default
    def _default_std_dev(self) -> float:
        return self.ground_state_energy / 2 / np.sqrt(2 * self.dimension)

    _nickname: str = field(init=False, default="BdGC", repr=False)

    @property
    def _path_name(self) -> str:
        return "BdG_C"

    @property
    def _latex_name(self) -> str:
        return "\\textrm{{BdG(C)}}"

    def generate_matrix(self, use_complex_dtype: bool = True) -> np.ndarray:
        complex_dtype: type[np.complexfloating] = self.complex_dtype.type
        real_dtype: type[np.floating] = self.real_dtype.type
        rng: np.random.Generator = self.rng
        size: int = self.dimension
        std_dev: float = self.std_dev

        matrix: np.ndarray = np.empty((size, size), complex_dtype, order="F")
        _create_bdgc_matrix(matrix, rng, real_dtype, std_dev)
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
            _create_bdgc_matrix(matrix, rng, real_dtype, std_dev)
            yield matrix
