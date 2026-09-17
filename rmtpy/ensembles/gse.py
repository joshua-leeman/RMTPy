from collections.abc import Iterator
from typing import ClassVar, cast, override

import attrs
import numba
import numpy as np

from .many_body_ensemble import HermitianMatrix
from .wigner_dyson_ensemble import WignerDysonEnsemble

INITIALISM: str = "GSE"

DYSON_INDEX: int = 4


def _build_gse_matrix(
    matrix: HermitianMatrix,
    real_dtype: type[np.float64],
    std_dev: float,
    rng: np.random.Generator,
) -> None:
    halfway = cast(int, matrix.shape[0] // 2)
    top_left_block = matrix[:halfway, :halfway]
    top_right_block = matrix[:halfway, halfway:]
    bottom_left_block = matrix[halfway:, :halfway]
    bottom_right_block = matrix[halfway:, halfway:]

    _build_gue_matrix(top_left_block, real_dtype, std_dev, rng)
    _ = np.conjugate(top_left_block, out=bottom_right_block)

    _build_skew_symmetric_matrix(top_right_block, real_dtype, std_dev, rng)
    _ = np.negative(top_right_block, out=bottom_left_block)
    _ = np.conjugate(bottom_left_block, out=bottom_left_block)


@numba.njit(boundscheck=False, cache=True, fastmath=True)
def _build_gue_matrix(
    matrix: HermitianMatrix,
    real_dtype: type[np.float64],
    std_dev: float,
    rng: np.random.Generator,
) -> None:
    size = cast(int, matrix.shape[0])
    for i in range(size):
        matrix[i, i] = 2 * std_dev * rng.standard_normal(None, real_dtype)
        matrix[i + 1 :, i] = std_dev * (
            rng.standard_normal(size - i - 1, real_dtype)
            + 1j * rng.standard_normal(size - i - 1, real_dtype)
        )
        matrix[i, i + 1 :] = np.conj(matrix[i + 1 :, i])


@numba.njit(boundscheck=False, cache=True, fastmath=True)
def _build_skew_symmetric_matrix(
    matrix: HermitianMatrix,
    real_dtype: type[np.float64],
    std_dev: float,
    rng: np.random.Generator,
) -> None:
    size = cast(int, matrix.shape[0])
    for i in range(size):
        matrix[i, i] = 0.0
        matrix[i + 1 :, i] = std_dev * (
            rng.standard_normal(size - 1 - i, real_dtype)
            + 1j * rng.standard_normal(size - 1 - i, real_dtype)
        )
        matrix[i, i + 1 :] = -matrix[i + 1 :, i]


def _compute_standard_deviation(gse: GaussianSymplecticEnsemble) -> float:
    return cast(float, gse.spectral_radius / 2 / np.sqrt(2 * gse.dimension))


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class GaussianSymplecticEnsemble(WignerDysonEnsemble):
    initialism: ClassVar[str] = INITIALISM

    std_dev: float = attrs.field(
        default=attrs.Factory(_compute_standard_deviation, takes_self=True),
        init=False,
        repr=False,
    )
    dyson_index: int = attrs.field(
        default=DYSON_INDEX,
        init=False,
        repr=False,
    )

    @override
    def generate_matrix(
        self,
        *,
        use_complex_dtype: bool = False,
    ) -> HermitianMatrix:
        matrix = self._allocate_complex_hermitian_matrix_memory()
        _build_gse_matrix(matrix, self.real_dtype.type, self.std_dev, self.rng)
        return matrix

    @override
    def matrix_stream(
        self,
        realizs: int,
        *,
        use_complex_dtype: bool = False,
    ) -> Iterator[HermitianMatrix]:
        matrix = self._allocate_complex_hermitian_matrix_memory()
        for _ in range(realizs):
            _build_gse_matrix(matrix, self.real_dtype.type, self.std_dev, self.rng)
            yield matrix
