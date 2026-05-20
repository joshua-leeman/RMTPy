from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Iterator
from typing import ClassVar

import attrs
import numpy as np
from scipy.interpolate import PchipInterpolator

# BLAS/LAPACK routines by dtype: s=float32, d=float64, c=complex64, z=complex128
from scipy.linalg.blas import ccopy, dcopy, scopy, zcopy
from scipy.linalg.blas import cgemm, dgemm, sgemm, zgemm
from scipy.linalg.blas import cher, dsyr, ssyr, zher
from scipy.linalg.lapack import cgeev, dgeev, sgeev, zgeev
from scipy.linalg.lapack import cheev, dsyev, ssyev, zheev

import rmtpy.density
import rmtpy.universal
import rmtpy.validators
from .base import RandomMatrixEnsemble

INITIALISM: str = "MBE"

INTERACTION_STRENGTH_DEFAULT: float = 1.0
INTERACTION_STRENGTH_METADATA: dict[str, str] = {
    "dir_name": "J",
}
DYSON_INDEX: int = 0
NUM_MAJORANAS_MINIMUM: int = 4
NUM_MAJORANAS_MAXIMUM: int = 32
NUM_MAJORANAS_METADATA: dict[str, str] = {
    "dir_name": "Nm",
    "latex_name": r"N_\textrm{\tiny m}",
}


def compute_dimension(mbe: ManyBodyEnsemble) -> int:
    return 2 ** (mbe.num_majoranas // 2 - 1)


def compute_spectral_radius(mbe: ManyBodyEnsemble) -> float:
    return mbe.num_majoranas * mbe.interaction_strength


def create_spectral_model(mbe: ManyBodyEnsemble) -> rmtpy.density.DensityModel:
    return rmtpy.density.DensityModel(
        dimension=mbe.dimension,
        support=tuple([-mbe.spectral_radius, mbe.spectral_radius]),
        polynomials=mbe.spectral_polynomials,
        max_polynomial_degree=mbe.max_spectral_polynomial_degree,
        weight_function=mbe.spectral_weight,
        sample_stream=mbe.eigvals_stream,
    )


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class ManyBodyEnsemble(RandomMatrixEnsemble):
    initialism: ClassVar[str] = INITIALISM

    num_majoranas: int = attrs.field(
        validator=[
            attrs.validators.instance_of(int),
            attrs.validators.ge(NUM_MAJORANAS_MINIMUM),
            attrs.validators.le(NUM_MAJORANAS_MAXIMUM),
            rmtpy.validators.validate_even_number,
        ],
        metadata=NUM_MAJORANAS_METADATA,
    )
    interaction_strength: float = attrs.field(
        default=INTERACTION_STRENGTH_DEFAULT,
        converter=float,
        validator=attrs.validators.gt(0.0),
        metadata=INTERACTION_STRENGTH_METADATA,
    )
    max_spectral_polynomial_degree: int = attrs.field(
        default=rmtpy.density.MAX_POLYNOMIAL_DEGREE_DEFAULT,
        converter=int,
        validator=attrs.validators.gt(0),
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

    spectral_polynomials: Callable[[np.ndarray, int], np.ndarray] = attrs.field(
        default=None,
        init=False,
        repr=False,
    )
    spectral_weight: Callable[[np.ndarray], np.ndarray] = attrs.field(
        default=None,
        init=False,
        repr=False,
    )
    spectral_density: rmtpy.density.DensityModel = attrs.field(
        default=attrs.Factory(create_spectral_model, takes_self=True),
        init=False,
        repr=False,
    )

    @property
    def eigval_degeneracy(self) -> int:
        return rmtpy.universal.eigval_degeneracy(self.dyson_index)

    @property
    def universality_class(self) -> str | None:
        return rmtpy.universal.universality_class(self.dyson_index)

    @abstractmethod
    def generate_matrix(self, use_complex_dtype: bool = False) -> None:
        raise NotImplementedError()

    @abstractmethod
    def matrix_stream(self, realizs: int, use_complex_dtype: bool = False) -> None:
        raise NotImplementedError()

    def average_spectral_pdf(self, energies: np.ndarray) -> np.ndarray:
        return self.spectral_density.average_pdf(energies)

    def average_spectral_cdf(self, energies: np.ndarray) -> np.ndarray:
        return self.spectral_density.average_cdf(energies)

    def compute_variate_spectral_coeffs(self, eigvals: np.ndarray) -> np.ndarray:
        return self.spectral_density.compute_variate_coeffs(eigvals)

    def create_variate_spectral_cdf_interpolator(
        self,
        interval: tuple[float, float] | None = None,
        spectral_coeffs: np.ndarray | None = None,
        eigvals: np.ndarray | None = None,
    ) -> PchipInterpolator:
        return self.spectral_density.create_variate_cdf_interpolator(
            interval=interval, coeffs=spectral_coeffs, sample=eigvals
        )

    def variate_spectral_pdf(
        self,
        energies: np.ndarray,
        spectral_coeffs: np.ndarray | None = None,
        eigvals: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.spectral_density.variate_pdf(
            energies, coeffs=spectral_coeffs, sample=eigvals
        )

    def variate_spectral_cdf(
        self,
        energies: np.ndarray,
        spectral_coeffs: np.ndarray | None = None,
        eigvals: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.spectral_density.variate_cdf(
            energies, coeffs=spectral_coeffs, sample=eigvals
        )

    def eigsys_stream(
        self, realizs: int, use_complex_dtype: bool = False
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        lapack_heev: type = self._pick_lapack_heev(use_complex_dtype)
        for matrix in self.matrix_stream(realizs, use_complex_dtype):
            eigvals, eigvecs, _ = lapack_heev(matrix, compute_v=1, overwrite_a=True)
            yield eigvals, eigvecs

    def eigvals_stream(
        self, realizs: int, use_complex_dtype: bool = False
    ) -> Iterator[np.ndarray]:
        lapack_heev: type = self._pick_lapack_heev(use_complex_dtype)
        for matrix in self.matrix_stream(realizs, use_complex_dtype):
            eigvals = lapack_heev(matrix, compute_v=0, overwrite_a=True)[0]
            yield eigvals

    def porter_thomas_distribution(
        self, num_channels: int, widths: np.ndarray
    ) -> np.ndarray:
        return rmtpy.universal.porter_thomas_distribution(
            self.dyson_index, num_channels, widths
        )

    def wigner_surmise(self, spacings: np.ndarray) -> np.ndarray:
        return rmtpy.universal.wigner_surmise(self.dyson_index, spacings)

    def universal_csff(self, times: np.ndarray) -> np.ndarray:
        return rmtpy.universal.universal_csff(self.dyson_index, self.dimension, times)

    def _initialize_matrix(self, use_complex_dtype: bool = False) -> np.ndarray:
        size: int = self.dimension
        if use_complex_dtype or self.dyson_index != 1:
            return np.empty((size, size), self.complex_dtype.type, order="F")
        else:
            return np.empty((size, size), self.real_dtype.type, order="F")

    def _pick_blas_copy(self, use_complex_dtype: bool) -> type:
        if use_complex_dtype or self.dyson_index != 1:
            if self.complex_dtype.type == np.complex64:
                return ccopy
            else:
                return zcopy
        else:
            if self.real_dtype.type == np.float32:
                return scopy
            else:
                return dcopy

    def _pick_blas_gemm(self, use_complex_dtype: bool) -> type:
        if use_complex_dtype or self.dyson_index != 1:
            if self.complex_dtype.type == np.complex64:
                return cgemm
            else:
                return zgemm
        else:
            if self.real_dtype.type == np.float32:
                return sgemm
            else:
                return dgemm

    def _pick_blas_her(self, use_complex_dtype: bool) -> type:
        if use_complex_dtype or self.dyson_index != 1:
            if self.complex_dtype.type == np.complex64:
                return cher
            else:
                return zher
        else:
            if self.real_dtype.type == np.float32:
                return ssyr
            else:
                return dsyr

    def _pick_lapack_geev(self, use_complex_dtype: bool) -> type:
        if use_complex_dtype or self.dyson_index != 1:
            if self.complex_dtype.type == np.complex64:
                return cgeev
            else:
                return zgeev
        else:
            if self.real_dtype.type == np.float32:
                return sgeev
            else:
                return dgeev

    def _pick_lapack_heev(self, use_complex_dtype: bool) -> type:
        if use_complex_dtype or self.dyson_index != 1:
            if self.complex_dtype.type == np.complex64:
                return cheev
            else:
                return zheev
        else:
            if self.real_dtype.type == np.float32:
                return ssyev
            else:
                return dsyev
