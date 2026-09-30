from abc import ABC, abstractmethod
from collections.abc import Iterator
from functools import cached_property
from typing import ClassVar, cast, override

import attrs
import numpy as np
import scipy.linalg.blas
import scipy.linalg.lapack

from ..conversion import RMT_CONVERTER, SourceDict
from ..density import MAX_POLYNOMIAL_DEGREE, DensityModel
from ..polynomials import FloatFunction, OrthogonalPolynomials
from ..universal import (
    connected_sff,
    eigval_degeneracy,
    porter_thomas_distribution,
    universality_class,
    wigner_surmise,
)
from ..validators import is_even_number
from .base_ensemble import RandomMatrixEnsemble

type RealSymmetricMatrix = np.ndarray[tuple[int, int], np.dtype[np.floating]]
type HermitianMatrix = np.ndarray[tuple[int, int], np.dtype[np.complexfloating]]
type RealEigenvalues = np.ndarray[tuple[int], np.dtype[np.floating]]
type OrthogonalMatrix = np.ndarray[tuple[int, int], np.dtype[np.floating]]
type UnitaryMatrix = np.ndarray[tuple[int, int], np.dtype[np.complexfloating]]

INITIALISM: str = "MBE"

DYSON_INDEX: int = 0

NUM_MAJORANAS_MIN: int = 4
NUM_MAJORANAS_MAX: int = 32
NUM_MAJORANAS_METADATA: dict[str, str] = {
    "dir_name": "Nm",
}

INTERACTION_STRENGTH: float = 1.0
INTERACTION_STRENGTH_METADATA: dict[str, str] = {
    "dir_name": "J",
}

MAX_SPECTRAL_POLYNOMIAL_DEGREE_METADATA: dict[str, str] = {
    "dir_name": "max_polydeg",
}


def _compute_dimension(mbe: ManyBodyEnsemble) -> int:
    return cast(int, pow(2, mbe.num_majoranas // 2 - 1))


def _compute_spectral_radius(mbe: ManyBodyEnsemble) -> float:
    return mbe.num_majoranas * mbe.interaction_strength


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class ManyBodyEnsemble(RandomMatrixEnsemble, ABC):
    initialism: ClassVar[str] = INITIALISM

    num_majoranas: int = attrs.field(
        validator=(
            attrs.validators.instance_of(int),
            attrs.validators.ge(NUM_MAJORANAS_MIN),
            attrs.validators.le(NUM_MAJORANAS_MAX),
            is_even_number,
        ),
        metadata=NUM_MAJORANAS_METADATA,
    )
    interaction_strength: float = attrs.field(
        default=INTERACTION_STRENGTH,
        converter=float,
        validator=attrs.validators.gt(0.0),
        metadata=INTERACTION_STRENGTH_METADATA,
    )
    max_spectral_polynomial_degree: int = attrs.field(
        default=MAX_POLYNOMIAL_DEGREE,
        converter=int,
        validator=attrs.validators.ge(0),
        metadata=MAX_SPECTRAL_POLYNOMIAL_DEGREE_METADATA,
    )

    dyson_index: int = attrs.field(
        default=DYSON_INDEX,
        init=False,
        repr=False,
    )
    dimension: int = attrs.field(
        default=attrs.Factory(_compute_dimension, takes_self=True),
        init=False,
    )
    spectral_radius: float = attrs.field(
        default=attrs.Factory(_compute_spectral_radius, takes_self=True),
        init=False,
        repr=False,
    )

    @classmethod
    @override
    def create(cls, src: SourceDict | RandomMatrixEnsemble) -> ManyBodyEnsemble:
        return RMT_CONVERTER.structure(src, cls)

    @property
    def eigval_degeneracy(self) -> int:
        return eigval_degeneracy(dyson_index=self.dyson_index)

    @property
    def universality_class(self) -> str | None:
        return universality_class(dyson_index=self.dyson_index)

    @property
    @override
    def _latex_name(self) -> str:
        return rf"{{{super()._latex_name}}}(N_\text{{m}} = {{{self.num_majoranas}}})"

    @cached_property
    def spectral_density(self) -> DensityModel:
        return DensityModel(
            max_polynomial_degree=self.max_spectral_polynomial_degree,
            polynomials=self.assign_spectral_polynomials(),
            weight_function=self.assign_spectral_weight(),
            dimension=self.dimension,
            support=(-self.spectral_radius, self.spectral_radius),
            sample_stream=self.eigvals_stream,
        )

    @abstractmethod
    def generate_matrix(
        self,
        *,
        use_complex_dtype: bool = False,
    ) -> RealSymmetricMatrix | HermitianMatrix:
        raise NotImplementedError()

    @abstractmethod
    def matrix_stream(
        self,
        realizs: int,
        *,
        use_complex_dtype: bool = False,
    ) -> Iterator[RealSymmetricMatrix | HermitianMatrix]:
        raise NotImplementedError()

    def eigsys_stream(
        self,
        realizs: int,
        *,
        use_complex_dtype: bool = False,
    ) -> Iterator[tuple[RealEigenvalues, OrthogonalMatrix | UnitaryMatrix]]:
        lapack_heev = self._pick_lapack_heev(use_complex_dtype=use_complex_dtype)
        for matrix in self.matrix_stream(realizs, use_complex_dtype=use_complex_dtype):
            eigvals, eigvecs, _ = cast(
                tuple[RealEigenvalues, OrthogonalMatrix | UnitaryMatrix, object],
                lapack_heev(matrix, compute_v=1, overwrite_a=True),
            )
            yield eigvals, eigvecs

    def eigvals_stream(
        self,
        realizs: int,
        *,
        use_complex_dtype: bool = False,
    ) -> Iterator[RealEigenvalues]:
        lapack_heev = self._pick_lapack_heev(use_complex_dtype=use_complex_dtype)
        for matrix in self.matrix_stream(realizs, use_complex_dtype=use_complex_dtype):
            eigvals, _, _ = cast(
                tuple[RealEigenvalues, object, object],
                lapack_heev(matrix, compute_v=0, overwrite_a=True),
            )
            yield eigvals

    def porter_thomas_distribution(
        self,
        widths: np.ndarray[tuple[int], np.dtype[np.floating]],
        /,
        *,
        num_channels: int = 1,
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        return porter_thomas_distribution(
            widths,
            dyson_index=self.dyson_index,
            num_channels=num_channels,
        )

    def wigner_surmise(
        self,
        spacings: np.ndarray[tuple[int], np.dtype[np.floating]],
        /,
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        return wigner_surmise(spacings, dyson_index=self.dyson_index)

    def universal_connected_sff(
        self,
        /,
        times: np.ndarray[tuple[int], np.dtype[np.floating]],
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        return connected_sff(
            times,
            dyson_index=self.dyson_index,
            dimension=self.dimension,
        )

    def _allocate_complex_hermitian_matrix_memory(self) -> HermitianMatrix:
        return np.empty((self.dimension, self.dimension), self.complex_dtype, order="F")

    def _allocate_empty_real_symmetric_matrix_memory(self) -> RealSymmetricMatrix:
        return np.empty((self.dimension, self.dimension), self.real_dtype, order="F")

    def assign_spectral_polynomials(self) -> OrthogonalPolynomials | None:
        return

    def assign_spectral_weight(self) -> FloatFunction | None:
        return

    def _pick_linalg_dtype(self, use_complex_dtype: bool = False) -> np.dtype:
        if use_complex_dtype or self.dyson_index != 1:
            return self.complex_dtype
        else:
            return self.real_dtype

    def _pick_blas_gemm(self, *, use_complex_dtype: bool = False):
        matrix_dtype = self._pick_linalg_dtype(use_complex_dtype=use_complex_dtype)
        return scipy.linalg.blas.get_blas_funcs("gemm", dtype=matrix_dtype)

    def _pick_blas_her(self, *, use_complex_dtype: bool = False):
        matrix_dtype = self._pick_linalg_dtype(use_complex_dtype=use_complex_dtype)
        routine = "her" if np.issubdtype(matrix_dtype, np.complexfloating) else "syr"
        return scipy.linalg.blas.get_blas_funcs(routine, dtype=matrix_dtype)

    def _pick_lapack_geev(self, *, use_complex_dtype: bool = False):
        matrix_dtype = self._pick_linalg_dtype(use_complex_dtype=use_complex_dtype)
        return scipy.linalg.lapack.get_lapack_funcs("geev", dtype=matrix_dtype)

    def _pick_lapack_heev(self, *, use_complex_dtype: bool = False):
        matrix_dtype = self._pick_linalg_dtype(use_complex_dtype=use_complex_dtype)
        routine = "heev" if np.issubdtype(matrix_dtype, np.complexfloating) else "syev"
        return scipy.linalg.lapack.get_lapack_funcs(routine, dtype=matrix_dtype)
