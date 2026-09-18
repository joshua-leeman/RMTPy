from collections.abc import Iterator
from typing import ClassVar, cast, override

import attrs
import numba
import numpy as np

from .many_body_ensemble import HermitianMatrix
from .wigner_dyson_ensemble import WignerDysonEnsemble

INITIALISM: str = "GUE"

DYSON_INDEX: int = 2


@numba.njit(boundscheck=False, cache=True, fastmath=True)
def _build_gue_matrix(
    matrix: HermitianMatrix,
    real_dtype: type[np.floating],
    std_dev: float,
    rng: np.random.Generator,
) -> None:
    size = matrix.shape[0]  # pyright: ignore[reportAny]
    for i in range(size):  # pyright: ignore[reportAny]
        matrix[i, i] = 2 * std_dev * rng.standard_normal(None, real_dtype)
        matrix[i + 1 :, i] = std_dev * (
            rng.standard_normal(size - i - 1, real_dtype)  # pyright: ignore[reportAny]
            + 1j * rng.standard_normal(size - i - 1, real_dtype)  # pyright: ignore[reportAny]
        )
        matrix[i, i + 1 :] = np.conj(matrix[i + 1 :, i])


def _compute_standard_deviation(gue: GaussianUnitaryEnsemble) -> float:
    return cast(float, gue.spectral_radius / 2 / np.sqrt(2 * gue.dimension))


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class GaussianUnitaryEnsemble(WignerDysonEnsemble):
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
        _build_gue_matrix(matrix, self.real_dtype.type, self.std_dev, self.rng)
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
            _build_gue_matrix(matrix, self.real_dtype.type, self.std_dev, self.rng)
            yield matrix
