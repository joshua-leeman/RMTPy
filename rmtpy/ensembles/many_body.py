from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Iterator
from typing import Any, ClassVar

import attrs
import numpy as np
import scipy.linalg.blas
import scipy.linalg.lapack

import rmtpy.density
import rmtpy.universal
import rmtpy.validators

from .base import RandomMatrixEnsemble

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


def compute_dimension(mbe: ManyBodyEnsemble) -> int:
    return 2 ** (mbe.num_majoranas // 2 - 1)


def compute_spectral_radius(mbe: ManyBodyEnsemble) -> float:
    return mbe.num_majoranas * mbe.interaction_strength


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class ManyBodyEnsemble(RandomMatrixEnsemble):
    """Closed Hamiltonian ensemble with spectral-density and universal-law helpers."""

    initialism: ClassVar[str] = INITIALISM

    num_majoranas: int = attrs.field(
        validator=[
            attrs.validators.instance_of(int),
            attrs.validators.ge(NUM_MAJORANAS_MIN),
            attrs.validators.le(NUM_MAJORANAS_MAX),
            rmtpy.validators.is_even_number,
        ],
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

    dimension: int = attrs.field(
        default=attrs.Factory(compute_dimension, takes_self=True),
        init=False,
    )
    spectral_radius: float = attrs.field(
        default=attrs.Factory(compute_spectral_radius, takes_self=True),
        init=False,
        repr=False,
    )
    dyson_index: int | float = attrs.field(
        default=DYSON_INDEX,
        init=False,
        repr=False,
    )

    spectral_polynomials: None = attrs.field(
        default=None,
        init=False,
        repr=False,
    )
    spectral_weight: None = attrs.field(
        default=None,
        init=False,
        repr=False,
    )
    spectral_density: rmtpy.density.DensityModel = attrs.field(
        default=None,
        init=False,
        repr=False,
    )

    def __attrs_post_init__(self) -> None:
        spectral_density = rmtpy.density.DensityModel(
            dimension=self.dimension,
            support=(-self.spectral_radius, self.spectral_radius),
            polynomials=self.spectral_polynomials,
            max_polynomial_degree=self.max_spectral_polynomial_degree,
            weight_function=self.spectral_weight,
            sample_stream=self.eigvals_stream,
        )
        object.__setattr__(self, "spectral_density", spectral_density)

    @property
    def eigval_degeneracy(self) -> int:
        return rmtpy.universal.eigval_degeneracy(dyson_index=self.dyson_index)

    @property
    def latex_name(self) -> str:
        return rf"{{{super().latex_name}}}({{{self.num_majoranas}}})"

    @property
    def universality_class(self) -> str | None:
        return rmtpy.universal.universality_class(dyson_index=self.dyson_index)

    @abstractmethod
    def generate_matrix(self, *, use_complex_dtype: bool = False) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def matrix_stream(
        self, realizs: int, *, use_complex_dtype: bool = False
    ) -> Iterator[np.ndarray]:
        raise NotImplementedError()

    def eigsys_stream(
        self, realizs: int, *, use_complex_dtype: bool = False
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        lapack_heev = self._pick_lapack_heev(use_complex_dtype=use_complex_dtype)

        for matrix in self.matrix_stream(realizs, use_complex_dtype=use_complex_dtype):
            eigvals, eigvecs, _ = lapack_heev(matrix, compute_v=1, overwrite_a=True)
            yield eigvals, eigvecs

    def eigvals_stream(
        self, realizs: int, *, use_complex_dtype: bool = False
    ) -> Iterator[np.ndarray]:
        lapack_heev = self._pick_lapack_heev(use_complex_dtype=use_complex_dtype)

        for matrix in self.matrix_stream(realizs, use_complex_dtype=use_complex_dtype):
            yield lapack_heev(matrix, compute_v=0, overwrite_a=True)[0]

    def porter_thomas_distribution(
        self, widths: np.ndarray, *, num_channels: int = 1
    ) -> np.ndarray:
        return rmtpy.universal.porter_thomas_distribution(
            widths, dyson_index=self.dyson_index, num_channels=num_channels
        )

    def wigner_surmise(self, spacings: np.ndarray) -> np.ndarray:
        return rmtpy.universal.wigner_surmise(spacings, dyson_index=self.dyson_index)

    def universal_connected_sff(self, times: np.ndarray) -> np.ndarray:
        return rmtpy.universal.connected_sff(
            times, dyson_index=self.dyson_index, dimension=self.dimension
        )

    def _empty_matrix(self, *, use_complex_dtype: bool = False) -> np.ndarray:
        if use_complex_dtype or self.dyson_index != 1:
            return np.empty(
                (self.dimension, self.dimension), self.complex_dtype.type, order="F"
            )
        else:
            return np.empty(
                (self.dimension, self.dimension), self.real_dtype.type, order="F"
            )

    def _pick_blas_gemm(self, *, use_complex_dtype: bool = False) -> Callable[..., Any]:
        if use_complex_dtype or self.dyson_index != 1:
            if self.complex_dtype.type == np.complex64:
                return scipy.linalg.blas.cgemm
            else:
                return scipy.linalg.blas.zgemm
        else:
            if self.real_dtype.type == np.float32:
                return scipy.linalg.blas.sgemm
            else:
                return scipy.linalg.blas.dgemm

    def _pick_blas_her(self, *, use_complex_dtype: bool = False) -> Callable[..., Any]:
        if use_complex_dtype or self.dyson_index != 1:
            if self.complex_dtype.type == np.complex64:
                return scipy.linalg.blas.cher
            else:
                return scipy.linalg.blas.zher
        else:
            if self.real_dtype.type == np.float32:
                return scipy.linalg.blas.ssyr
            else:
                return scipy.linalg.blas.dsyr

    def _pick_lapack_geev(self, *, use_complex_dtype: bool = False) -> Callable[..., Any]:
        if use_complex_dtype or self.dyson_index != 1:
            if self.complex_dtype.type == np.complex64:
                return scipy.linalg.lapack.cgeev
            else:
                return scipy.linalg.lapack.zgeev
        else:
            if self.real_dtype.type == np.float32:
                return scipy.linalg.lapack.sgeev
            else:
                return scipy.linalg.lapack.dgeev

    def _pick_lapack_heev(self, *, use_complex_dtype: bool = False) -> Callable[..., Any]:
        if use_complex_dtype or self.dyson_index != 1:
            if self.complex_dtype.type == np.complex64:
                return scipy.linalg.lapack.cheev
            else:
                return scipy.linalg.lapack.zheev
        else:
            if self.real_dtype.type == np.float32:
                return scipy.linalg.lapack.ssyev
            else:
                return scipy.linalg.lapack.dsyev
