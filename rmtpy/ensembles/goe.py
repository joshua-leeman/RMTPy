from collections.abc import Iterator
from typing import ClassVar, cast, override

import attrs
import numba
import numpy as np

from .many_body_ensemble import HermitianMatrix, RealSymmetricMatrix
from .wigner_dyson_ensemble import WignerDysonEnsemble

INITIALISM: str = "GOE"

DYSON_INDEX: int = 1


@numba.njit(boundscheck=False, cache=True, fastmath=True)
def build_goe_matrix(
    matrix: RealSymmetricMatrix | HermitianMatrix,
    real_dtype: type[np.floating],
    std_dev: float,
    rng: np.random.Generator,
) -> None:
    size = matrix.shape[0]
    for i in range(size):
        matrix[i, i] = 2 * std_dev * rng.standard_normal(None, real_dtype)
        matrix[i + 1 :, i] = std_dev * rng.standard_normal(size - i - 1, real_dtype)
        matrix[i, i + 1 :] = matrix[i + 1 :, i]


def _compute_standard_deviation(goe: GaussianOrthogonalEnsemble) -> float:
    return goe.spectral_radius / 2 / cast(float, np.sqrt(goe.dimension))


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class GaussianOrthogonalEnsemble(WignerDysonEnsemble):
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
    ) -> RealSymmetricMatrix | HermitianMatrix:
        if use_complex_dtype:
            matrix = self._allocate_complex_hermitian_matrix_memory()
        else:
            matrix = self._allocate_empty_real_symmetric_matrix_memory()

        build_goe_matrix(matrix, self.real_dtype.type, self.std_dev, self.rng)
        return matrix

    @override
    def matrix_stream(
        self,
        realizs: int,
        *,
        use_complex_dtype: bool = False,
    ) -> Iterator[RealSymmetricMatrix | HermitianMatrix]:
        if use_complex_dtype:
            matrix = self._allocate_complex_hermitian_matrix_memory()
        else:
            matrix = self._allocate_empty_real_symmetric_matrix_memory()

        for _ in range(realizs):
            build_goe_matrix(matrix, self.real_dtype.type, self.std_dev, self.rng)
            yield matrix
