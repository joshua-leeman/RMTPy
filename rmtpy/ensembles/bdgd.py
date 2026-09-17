from collections.abc import Iterator
from typing import ClassVar, cast, override

import attrs
import numba
import numpy as np

from .many_body_ensemble import HermitianMatrix
from .wigner_dyson_ensemble import WignerDysonEnsemble

INITIALISM: str = "BdGD"

LATEX_NAME: str = "\\text{{BdG(D)}}"

TOKEN_NAME: str = "BdG_D"

DYSON_INDEX: int = 2


@numba.njit(boundscheck=False, cache=True, fastmath=True)
def _build_bdgd_matrix(
    matrix: HermitianMatrix,
    real_dtype: type[np.float64],
    std_dev: float,
    rng: np.random.Generator,
) -> None:
    size = cast(int, matrix.shape[0])
    for i in range(size):
        matrix[i, i] = 0.0
        matrix[i + 1 :, i] = std_dev * 1j * rng.standard_normal(size - i - 1, real_dtype)
        matrix[i, i + 1 :] = np.conj(matrix[i + 1 :, i])


def _compute_standard_deviation(bdgd: BogoliubovDeGennesDEnsemble) -> float:
    return cast(float, bdgd.spectral_radius / 2 / np.sqrt(bdgd.dimension))


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class BogoliubovDeGennesDEnsemble(WignerDysonEnsemble):
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

    @property
    @override
    def latex_name(self) -> str:
        return rf"{{{LATEX_NAME}}}({{{self.num_majoranas}}})"

    @property
    @override
    def token_name(self) -> str:
        return TOKEN_NAME

    @override
    def generate_matrix(
        self,
        *,
        use_complex_dtype: bool = False,
    ) -> HermitianMatrix:
        matrix = self._allocate_complex_hermitian_matrix_memory()
        _build_bdgd_matrix(matrix, self.real_dtype.type, self.std_dev, self.rng)
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
            _build_bdgd_matrix(matrix, self.real_dtype.type, self.std_dev, self.rng)
            yield matrix
