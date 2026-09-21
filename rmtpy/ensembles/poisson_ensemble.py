from collections.abc import Callable, Iterator
from typing import ClassVar, cast, override

import attrs
import numba
import numpy as np
from numpy.typing import NDArray

from ..conversion import RMT_CONVERTER, SourceDict
from ..polynomials import (
    Float64Function,
    OrthogonalPolynomials,
    legendre_polynomial_weight,
    legendre_polynomials,
)
from ..universal import porter_thomas_distribution
from .base_ensemble import RandomMatrixEnsemble
from .many_body_ensemble import (
    HermitianMatrix,
    ManyBodyEnsemble,
    OrthogonalMatrix,
    RealEigenvalues,
    RealSymmetricMatrix,
    UnitaryMatrix,
)
from .wigner_dyson_ensemble import (
    WIGNER_DYSON_ENSEMBLE_INITIALISMS_BY_NAME,
    WIGNER_DYSON_ENSEMBLE_NAMES_BY_INITIALISM,
    WignerDysonEnsemble,
)

type UpperComplexTriangleCloner = Callable[[NDArray[np.complexfloating]], None]
type UpperRealTriangleCloner = Callable[[NDArray[np.floating]], None]

INITIALISM: str = "Poisson"

DYSON_INDEX: int = 0

eigvec_ensemble_FLAG: str = "GUE"
eigvec_ensemble_FLAG_METADATA: dict[str, str] = {
    "dir_name": "eigvecs",
}


def _cast_eigvec_ensemble_fag(flag: str) -> str:
    return WIGNER_DYSON_ENSEMBLE_INITIALISMS_BY_NAME.get(flag, flag)


def _compute_standard_deviation(poisson: PoissonEnsemble) -> float:
    return 2 * poisson.spectral_radius


@numba.njit(boundscheck=False, cache=True, fastmath=True)
def _symmetrize_upper_complex_triangle_to_lower(
    matrix: NDArray[np.complexfloating],
) -> None:
    size = matrix.shape[0]  # pyright: ignore[reportAny]
    for i in range(size):  # pyright: ignore[reportAny]
        matrix[i + 1 :, i] = matrix[i, i + 1 :].conj()


@numba.njit(boundscheck=False, cache=True, fastmath=True)
def _symmetrize_upper_real_triangle_to_lower(matrix: NDArray[np.floating]) -> None:
    size = matrix.shape[0]  # pyright: ignore[reportAny]
    for i in range(size):  # pyright: ignore[reportAny]
        matrix[i + 1 :, i] = matrix[i, i + 1 :]


def _instantiate_eigvec_ensemble(poisson: PoissonEnsemble) -> WignerDysonEnsemble:
    ensemble_dict = cast(SourceDict, RMT_CONVERTER.unstructure(poisson))
    flag = poisson.eigvec_ensemble_flag
    ensemble_dict["type"] = WIGNER_DYSON_ENSEMBLE_NAMES_BY_INITIALISM[flag]
    ensemble_parameters = cast(dict[str, object], ensemble_dict["parameters"])
    ensemble_parameters["seed"] = poisson.rng
    del ensemble_parameters["eigvec_ensemble_flag"]
    return RMT_CONVERTER.structure(ensemble_dict, WignerDysonEnsemble)


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class PoissonEnsemble(ManyBodyEnsemble):
    initialism: ClassVar[str] = INITIALISM

    eigvec_ensemble_flag: str = attrs.field(
        default=eigvec_ensemble_FLAG,
        converter=(str.lower, _cast_eigvec_ensemble_fag),
        validator=attrs.validators.in_(WIGNER_DYSON_ENSEMBLE_NAMES_BY_INITIALISM),
        metadata=eigvec_ensemble_FLAG_METADATA,
    )

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
    eigvec_ensemble: WignerDysonEnsemble = attrs.field(
        default=attrs.Factory(_instantiate_eigvec_ensemble, takes_self=True),
        init=False,
        repr=False,
    )

    @classmethod
    @override
    def create(
        cls,
        src: SourceDict | RandomMatrixEnsemble,
    ) -> PoissonEnsemble:
        return RMT_CONVERTER.structure(src, cls)

    @override
    def generate_matrix(
        self,
        *,
        use_complex_dtype: bool = False,
    ) -> RealSymmetricMatrix | HermitianMatrix:
        symmetrizer = self._pick_symmetrizer(use_complex_dtype=use_complex_dtype)

        lapack_heev = self._pick_lapack_heev(use_complex_dtype=use_complex_dtype)

        eigvecs = cast(
            NDArray[np.generic],
            lapack_heev(
                self.eigvec_ensemble.generate_matrix(
                    use_complex_dtype=use_complex_dtype,
                ),
                compute_v=1,
                overwrite_a=True,
            )[1],
        )
        eigvals = self.generate_eigenvalues()

        if use_complex_dtype or self.eigvec_ensemble.dyson_index != 1:
            matrix = self._allocate_complex_hermitian_matrix_memory()
        else:
            matrix = self._allocate_empty_real_symmetric_matrix_memory()
        matrix.fill(0.0)

        blas_her = self._pick_blas_her(use_complex_dtype=use_complex_dtype)
        for mu in range(self.dimension):
            blas_her(cast(float, eigvals[mu]), x=eigvecs[:, mu], a=matrix, overwrite_a=1)

        symmetrizer(matrix)

        return matrix

    @override
    def matrix_stream(
        self,
        realizs: int,
        *,
        use_complex_dtype: bool = False,
    ) -> Iterator[RealSymmetricMatrix | HermitianMatrix]:
        symmetrizer = self._pick_symmetrizer(use_complex_dtype=use_complex_dtype)

        if use_complex_dtype or self.eigvec_ensemble.dyson_index != 1:
            matrix = self._allocate_complex_hermitian_matrix_memory()
        else:
            matrix = self._allocate_empty_real_symmetric_matrix_memory()
        matrix.fill(0.0)

        blas_her = self._pick_blas_her(use_complex_dtype=use_complex_dtype)
        for eigvals, eigvecs in self.eigsys_stream(
            realizs,
            use_complex_dtype=use_complex_dtype,
        ):
            matrix.fill(0.0)
            for mu in range(self.dimension):
                blas_her(
                    cast(float, eigvals[mu]), x=eigvecs[:, mu], a=matrix, overwrite_a=1
                )

            symmetrizer(matrix)

            yield matrix

    @override
    def eigsys_stream(
        self,
        realizs: int,
        *,
        use_complex_dtype: bool = False,
    ) -> Iterator[tuple[RealEigenvalues, OrthogonalMatrix | UnitaryMatrix]]:
        for _, eigvecs in self.eigvec_ensemble.eigsys_stream(
            realizs, use_complex_dtype=use_complex_dtype
        ):
            yield self.generate_eigenvalues(), eigvecs

    @override
    def eigvals_stream(
        self,
        realizs: int,
        *,
        use_complex_dtype: bool = False,
    ) -> Iterator[RealEigenvalues]:
        for _ in range(realizs):
            yield self.generate_eigenvalues()

    @override
    def porter_thomas_distribution(
        self,
        widths: NDArray[np.float64],
        /,
        *,
        num_channels: int = 1,
    ) -> NDArray[np.float64]:
        return porter_thomas_distribution(
            widths,
            dyson_index=self.eigvec_ensemble.dyson_index,
            num_channels=num_channels,
        )

    @override
    def assign_spectral_polynomials(self) -> OrthogonalPolynomials:
        def poisson_spectral_polynomials(
            x: NDArray[np.float64],
            /,
            *,
            degree: int,
        ) -> NDArray[np.float64]:
            return legendre_polynomials(x, degree=degree)

        return poisson_spectral_polynomials

    @override
    def assign_spectral_weight(self) -> Float64Function:
        def poisson_spectral_weight(
            energies: NDArray[np.float64],
            /,
        ) -> NDArray[np.float64]:
            return legendre_polynomial_weight(
                energies,
                support_radius=self.spectral_radius,
            )

        return poisson_spectral_weight

    def generate_eigenvalues(self) -> RealEigenvalues:
        eigvals = self.rng.random(self.dimension, self.real_dtype.type)
        eigvals -= 0.5
        eigvals *= self.std_dev
        return np.sort(eigvals)

    @override
    def _pick_blas_gemm(self, *, use_complex_dtype: bool = False):
        return self.eigvec_ensemble._pick_blas_gemm(use_complex_dtype=use_complex_dtype)

    @override
    def _pick_blas_her(self, *, use_complex_dtype: bool = False):
        return self.eigvec_ensemble._pick_blas_her(use_complex_dtype=use_complex_dtype)

    @override
    def _pick_lapack_geev(self, *, use_complex_dtype: bool = False):
        return self.eigvec_ensemble._pick_lapack_geev(use_complex_dtype=use_complex_dtype)

    @override
    def _pick_lapack_heev(self, *, use_complex_dtype: bool = False):
        return self.eigvec_ensemble._pick_lapack_heev(use_complex_dtype=use_complex_dtype)

    def _pick_symmetrizer(
        self,
        *,
        use_complex_dtype: bool = False,
    ) -> Callable[..., None]:
        if use_complex_dtype or self.eigvec_ensemble.dyson_index != 1:
            return _symmetrize_upper_complex_triangle_to_lower
        else:
            return _symmetrize_upper_real_triangle_to_lower
