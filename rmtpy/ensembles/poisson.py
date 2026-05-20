from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, ClassVar

import attrs
import numba
import numpy as np

import rmtpy.conversion
import rmtpy.universal
from .many_body import ManyBodyEnsemble
from .wigner_dyson import (
    WIGNER_DYSON_ENSEMBLE_INITIALISMS_BY_NAME,
    WIGNER_DYSON_ENSEMBLE_NAMES_BY_INITIALISM,
    WignerDysonEnsemble,
)

INITIALISM: str = "Poisson"

EIGVECS_ENSEMBLE_FLAG_DEFAULT: str = "GUE"
DYSON_INDEX: int = 0


def compute_standard_deviation(poisson: PoissonEnsemble) -> float:
    return 2 * poisson.spectral_radius


@numba.njit(cache=True, fastmath=True)
def mirror_upper_to_lower_triangle_complex(matrix: np.ndarray) -> np.ndarray:
    size: int = matrix.shape[0]
    for i in range(size):
        matrix[i + 1 :, i] = matrix[i, i + 1 :].conj()


@numba.njit(cache=True, fastmath=True)
def mirror_upper_to_lower_triangle_real(matrix: np.ndarray) -> np.ndarray:
    size: int = matrix.shape[0]
    for i in range(size):
        matrix[i + 1 :, i] = matrix[i, i + 1 :]


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class PoissonEnsemble(ManyBodyEnsemble):
    initialism: ClassVar[str] = INITIALISM

    eigvecs_ensemble_flag: str = attrs.field(
        default=EIGVECS_ENSEMBLE_FLAG_DEFAULT,
        converter=[
            str.lower,
            lambda value: WIGNER_DYSON_ENSEMBLE_INITIALISMS_BY_NAME.get(value, value),
        ],
        validator=attrs.validators.in_(WIGNER_DYSON_ENSEMBLE_NAMES_BY_INITIALISM),
    )

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

    eigvecs_ensemble: WignerDysonEnsemble = attrs.field(init=False, repr=False)

    @eigvecs_ensemble.default
    def create_eigvecs_ensemble_instance(self) -> WignerDysonEnsemble:
        flag: str = self.eigvecs_ensemble_flag
        ens_dict: dict[str, Any] = rmtpy.conversion.CONVERTER.unstructure(self)
        ens_dict["name"] = WIGNER_DYSON_ENSEMBLE_NAMES_BY_INITIALISM[flag]
        return rmtpy.conversion.CONVERTER.structure(ens_dict, WignerDysonEnsemble)

    @property
    def path_name(self) -> str:
        return super().as_path + f"_{type(self.eigvecs_ensemble).initialism.lower()}"

    def generate_matrix(self, use_complex_dtype: bool = False) -> np.ndarray:
        matrix = self._initialize_matrix(use_complex_dtype)
        mirror_upper_triangle = self._pick_mirror_triangle_method(use_complex_dtype)

        eigvals: np.ndarray = self.rng.random(self.dimension, self.real_dtype.type)
        eigvals -= 0.5
        eigvals *= self.std_dev

        lapack_heev: type = self._pick_lapack_heev(use_complex_dtype)
        eigvecs: np.ndarray = lapack_heev(
            self.eigvecs_ensemble.generate_matrix(use_complex_dtype),
            compute_v=1,
            overwrite_a=True,
        )[1]

        blas_her: type = self._pick_blas_her(use_complex_dtype)
        for mu in range(self.dimension):
            blas_her(float(eigvals[mu]), x=eigvecs[:, mu], a=matrix, overwrite_a=1)
        mirror_upper_triangle(matrix)
        return matrix

    def matrix_stream(
        self, realizs: int, use_complex_dtype: bool = False
    ) -> Iterator[np.ndarray]:
        matrix = self._initialize_matrix(use_complex_dtype)
        mirror_upper_triangle = self._pick_mirror_triangle_method(use_complex_dtype)
        blas_her: type = self._pick_blas_her(use_complex_dtype)
        for eigvals, eigvecs in self.eigsys_stream(realizs, use_complex_dtype):
            for mu in range(self.dimension):
                blas_her(float(eigvals[mu]), x=eigvecs[:, mu], a=matrix, overwrite_a=1)
            mirror_upper_triangle(matrix)
            yield matrix

    def eigsys_stream(
        self, realizs: int, use_complex_dtype: bool = False
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for _, vecs in self.eigvecs_ensemble.eigsys_stream(realizs, use_complex_dtype):
            eigvals: np.ndarray = self.rng.random(self.dimension, self.real_dtype.type)
            eigvals -= 0.5
            eigvals *= self.std_dev
            yield np.sort(eigvals), vecs

    def eigvals_stream(
        self, realizs: int, use_complex_dtype: bool = False
    ) -> Iterator[np.ndarray]:
        for _ in range(realizs):
            eigvals: np.ndarray = self.rng.random(self.dimension, self.real_dtype.type)
            eigvals -= 0.5
            eigvals *= self.std_dev
            yield np.sort(eigvals)

    def spectral_pdf(self, eigvals: np.ndarray) -> np.ndarray:
        eigvals = np.asarray(eigvals)
        pdf: np.ndarray = np.zeros_like(eigvals, dtype=np.result_type(eigvals, float))
        pdf[np.abs(eigvals) < self.spectral_radius] = 1 / 2 / self.spectral_radius
        return pdf

    def cdf(self, eigvals: np.ndarray) -> np.ndarray:
        eigvals = np.asarray(eigvals)
        cdf: np.ndarray = np.zeros_like(eigvals, dtype=np.result_type(eigvals, float))
        mask: np.ndarray = np.abs(eigvals) < self.spectral_radius
        cdf[mask] = eigvals[mask] / (2 * self.spectral_radius) + 0.5
        cdf[eigvals > self.spectral_radius] = 1.0
        return cdf

    def porter_thomas_distribution(
        self, num_channels: int, widths: np.ndarray
    ) -> np.ndarray:
        return rmtpy.universal.porter_thomas_distribution(
            self.eigvecs_ensemble.dyson_index, num_channels, widths
        )

    def _initialize_matrix(self, use_complex_dtype: bool = False) -> np.ndarray:
        size: int = self.dimension
        if use_complex_dtype or self.eigvecs_ensemble.dyson_index != 1:
            return np.empty((size, size), self.complex_dtype.type, order="F")
        else:
            return np.empty((size, size), self.real_dtype.type, order="F")

    def _pick_blas_copy(self, use_complex_dtype: bool) -> type:
        return self.eigvecs_ensemble._pick_blas_copy(use_complex_dtype)

    def _pick_blas_gemm(self, use_complex_dtype: bool) -> type:
        return self.eigvecs_ensemble._pick_blas_gemm(use_complex_dtype)

    def _pick_blas_her(self, use_complex_dtype: bool) -> type:
        return self.eigvecs_ensemble._pick_blas_her(use_complex_dtype)

    def _pick_lapack_geev(self, use_complex_dtype: bool) -> type:
        return self.eigvecs_ensemble._pick_lapack_geev(use_complex_dtype)

    def _pick_lapack_heev(self, use_complex_dtype: bool) -> type:
        return self.eigvecs_ensemble._pick_lapack_heev(use_complex_dtype)

    def _pick_mirror_triangle_method(
        self, use_complex_dtype: bool = False
    ) -> Callable[[np.ndarray], np.ndarray]:
        if use_complex_dtype or self.eigvecs_ensemble.dyson_index != 1:
            return mirror_upper_to_lower_triangle_complex
        else:
            return mirror_upper_to_lower_triangle_real
