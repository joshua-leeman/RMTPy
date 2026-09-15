from collections.abc import Iterator
from typing import ClassVar, cast, override

import attrs
import numba
import numpy as np

from .many_body_ensemble import HermitianMatrix, RealSymmetricMatrix
from .wigner_dyson_ensemble import WignerDysonEnsemble

DYSON_INDEX: int = 1

INITIALISM: str = "GOE"


def compute_standard_deviation(goe: GaussianOrthogonalEnsemble) -> float:
    return cast(float, goe.spectral_radius / 2 / np.sqrt(goe.dimension))


@numba.njit(boundscheck=False, cache=True, fastmath=True)
def create_goe_matrix(
    matrix: RealSymmetricMatrix | HermitianMatrix,
    real_dtype: type[np.float64],
    std_dev: float,
    rng: np.random.Generator,
) -> None:
    size = cast(int, matrix.shape[0])
    for i in range(size):
        matrix[i, i] = 2 * std_dev * rng.standard_normal(None, real_dtype)
        matrix[i + 1 :, i] = std_dev * rng.standard_normal(size - i - 1, real_dtype)
        matrix[i, i + 1 :] = matrix[i + 1 :, i]


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class GaussianOrthogonalEnsemble(WignerDysonEnsemble):
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

    @override
    def generate_matrix(
        self,
        *,
        use_complex_dtype: bool = False,
    ) -> RealSymmetricMatrix | HermitianMatrix:
        matrix = self._create_empty_matrix(use_complex_dtype=use_complex_dtype)
        real_dtype = cast(type[np.float64], self.real_dtype.type)
        create_goe_matrix(matrix, real_dtype, self.std_dev, self.rng)
        return matrix

    @override
    def matrix_stream(
        self,
        realizs: int,
        *,
        use_complex_dtype: bool = False,
    ) -> Iterator[RealSymmetricMatrix | HermitianMatrix]:
        matrix = self._create_empty_matrix(use_complex_dtype=use_complex_dtype)
        real_dtype = cast(type[np.float64], self.real_dtype.type)
        for _ in range(realizs):
            create_goe_matrix(matrix, real_dtype, self.std_dev, self.rng)
            yield matrix
