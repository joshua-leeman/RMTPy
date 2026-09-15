from abc import ABC, abstractmethod
from collections.abc import Iterator
from functools import cached_property
from typing import ClassVar, cast, override

import attrs
import numpy as np
import scipy.linalg.blas
import scipy.linalg.lapack

import rmtpy.density
import rmtpy.universal
import rmtpy.validators
from rmtpy.density import DensityModel
from rmtpy.polynomials import OrthogonalPolynomials, RealFunction

from .base_ensemble import RandomMatrixEnsemble

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
            rmtpy.validators.is_even_number,
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
        default=rmtpy.density.MAX_POLYNOMIAL_DEGREE,
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

    @property
    def eigval_degeneracy(self) -> int:
        return rmtpy.universal.eigval_degeneracy(dyson_index=self.dyson_index)

    @property
    @override
    def latex_name(self) -> str:
        return rf"{{{super().latex_name}}}({{{self.num_majoranas}}})"

    @property
    def universality_class(self) -> str | None:
        return rmtpy.universal.universality_class(dyson_index=self.dyson_index)

    @cached_property
    def spectral_density(self) -> DensityModel:
        return DensityModel(
            max_polynomial_degree=self.max_spectral_polynomial_degree,
            polynomials=self._create_spectral_polynomials(),
            weight_function=self._create_spectral_weight(),
            dimension=self.dimension,
            support=(-self.spectral_radius, self.spectral_radius),
            sample_stream=self.eigvals_stream,
        )

    @abstractmethod
    def generate_matrix(self, *, use_complex_dtype: bool = False) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def matrix_stream(
        self, realizs: int, *, use_complex_dtype: bool = False
    ) -> Iterator[np.ndarray]:
        raise NotImplementedError()

    def eigsys_stream(
        self,
        realizs: int,
        *,
        use_complex_dtype: bool = False,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        lapack_heev = self._pick_lapack_heev(use_complex_dtype=use_complex_dtype)
        for matrix in self.matrix_stream(realizs, use_complex_dtype=use_complex_dtype):
            eigvals, eigvecs, _info = cast(
                tuple[np.ndarray, np.ndarray, object],
                lapack_heev(matrix, compute_v=1, overwrite_a=True),
            )
            yield eigvals, eigvecs

    def eigvals_stream(
        self,
        realizs: int,
        *,
        use_complex_dtype: bool = False,
    ) -> Iterator[np.ndarray]:
        lapack_heev = self._pick_lapack_heev(use_complex_dtype=use_complex_dtype)
        for matrix in self.matrix_stream(realizs, use_complex_dtype=use_complex_dtype):
            eigvals, _info = cast(
                tuple[np.ndarray, object],
                lapack_heev(matrix, compute_v=0, overwrite_a=True),
            )
            yield eigvals

    def porter_thomas_distribution(
        self,
        widths: np.ndarray,
        *,
        num_channels: int = 1,
    ) -> np.ndarray:
        return rmtpy.universal.porter_thomas_distribution(
            widths,
            dyson_index=self.dyson_index,
            num_channels=num_channels,
        )

    def wigner_surmise(self, spacings: np.ndarray) -> np.ndarray:
        return rmtpy.universal.wigner_surmise(spacings, dyson_index=self.dyson_index)

    def universal_connected_sff(self, times: np.ndarray) -> np.ndarray:
        return rmtpy.universal.connected_sff(
            times,
            dyson_index=self.dyson_index,
            dimension=self.dimension,
        )

    def _choose_linalg_dtype(self, use_complex_dtype: bool = False) -> np.dtype:
        if use_complex_dtype or self.dyson_index != 1:
            return self.complex_dtype

        return self.real_dtype

    def _create_empty_matrix(self, *, use_complex_dtype: bool = False) -> np.ndarray:
        matrix_dtype = self._choose_linalg_dtype(use_complex_dtype=use_complex_dtype)
        return np.empty((self.dimension, self.dimension), matrix_dtype, order="F")

    def _create_spectral_polynomials(self) -> OrthogonalPolynomials | None:
        return

    def _create_spectral_weight(self) -> RealFunction | None:
        return

    def _pick_blas_gemm(self, *, use_complex_dtype: bool = False):
        matrix_dtype = self._choose_linalg_dtype(use_complex_dtype=use_complex_dtype)
        return scipy.linalg.get_blas_funcs("gemm", dtype=matrix_dtype)

    def _pick_blas_her(self, *, use_complex_dtype: bool = False):
        matrix_dtype = self._choose_linalg_dtype(use_complex_dtype=use_complex_dtype)
        routine = "her" if np.issubdtype(matrix_dtype, np.complexfloating) else "syr"
        return scipy.linalg.blas.get_blas_funcs(routine, dtype=matrix_dtype)

    def _pick_lapack_geev(self, *, use_complex_dtype: bool = False):
        matrix_dtype = self._choose_linalg_dtype(use_complex_dtype=use_complex_dtype)
        return scipy.linalg.get_lapack_funcs("geev", dtype=matrix_dtype)

    def _pick_lapack_heev(self, *, use_complex_dtype: bool = False):
        matrix_dtype = self._choose_linalg_dtype(use_complex_dtype=use_complex_dtype)
        routine = "heev" if np.issubdtype(matrix_dtype, np.complexfloating) else "syev"
        return scipy.linalg.lapack.get_lapack_funcs(routine, dtype=matrix_dtype)
