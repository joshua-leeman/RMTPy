from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, ClassVar

import attrs
import numba
import numpy as np

import rmtpy.polynomials
import rmtpy.universal
from rmtpy.conversion import RMT_CONVERTER

from .many_body import ManyBodyEnsemble
from .wigner_dyson import (
    WIGNER_DYSON_ENSEMBLE_INITIALISMS_BY_NAME,
    WIGNER_DYSON_ENSEMBLE_NAMES_BY_INITIALISM,
    WignerDysonEnsemble,
)

INITIALISM: str = "Poisson"

DYSON_INDEX: int = 0

EIGVECS_ENSEMBLE_FLAG: str = "GUE"
EIGVECS_ENSEMBLE_FLAG_METADATA: dict[str, str] = {
    "dir_name": "eigvecs",
}


def compute_standard_deviation(poisson: PoissonEnsemble) -> float:
    return 2 * poisson.spectral_radius


def create_spectral_weight(
    poisson: PoissonEnsemble,
) -> Callable[[np.ndarray], np.ndarray]:
    def poisson_spectral_weight(energies: np.ndarray) -> np.ndarray:
        return rmtpy.polynomials.legendre_polynomial_weight_pdf(
            energies, radius=poisson.spectral_radius
        )

    return poisson_spectral_weight


@numba.njit(boundscheck=False, cache=True, fastmath=True)
def mirror_upper_to_lower_triangle_complex(matrix: np.ndarray) -> None:
    for i in range(matrix.shape[0]):
        matrix[i + 1 :, i] = matrix[i, i + 1 :].conj()


@numba.njit(boundscheck=False, cache=True, fastmath=True)
def mirror_upper_to_lower_triangle_real(matrix: np.ndarray) -> None:
    for i in range(matrix.shape[0]):
        matrix[i + 1 :, i] = matrix[i, i + 1 :]


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class PoissonEnsemble(ManyBodyEnsemble):
    initialism: ClassVar[str] = INITIALISM

    eigvecs_ensemble_flag: str = attrs.field(
        default=EIGVECS_ENSEMBLE_FLAG,
        converter=[
            str.lower,
            lambda value: WIGNER_DYSON_ENSEMBLE_INITIALISMS_BY_NAME.get(value, value),
        ],
        validator=attrs.validators.in_(WIGNER_DYSON_ENSEMBLE_NAMES_BY_INITIALISM),
        metadata=EIGVECS_ENSEMBLE_FLAG_METADATA,
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

    spectral_weight: Callable[[np.ndarray], np.ndarray] = attrs.field(
        default=attrs.Factory(create_spectral_weight, takes_self=True),
        init=False,
        repr=False,
    )
    spectral_polynomials: Callable[[np.ndarray], np.ndarray] = attrs.field(
        default=rmtpy.polynomials.legendre_polynomials,
        init=False,
        repr=False,
    )

    eigvecs_ensemble: WignerDysonEnsemble = attrs.field(init=False, repr=False)

    @eigvecs_ensemble.default
    def create_eigvecs_ensemble_instance(self) -> WignerDysonEnsemble:
        flag = self.eigvecs_ensemble_flag

        ens_dict = RMT_CONVERTER.unstructure(self)
        ens_dict["name"] = WIGNER_DYSON_ENSEMBLE_NAMES_BY_INITIALISM[flag]
        ens_dict["args"]["seed"] = self.rng
        ens_dict.pop("rng_state", None)
        return RMT_CONVERTER.structure(ens_dict, WignerDysonEnsemble)

    def generate_eigenvalues(self) -> np.ndarray:
        eigvals = self.rng.random(self.dimension, self.real_dtype.type)
        eigvals -= 0.5
        eigvals *= self.std_dev
        return np.sort(eigvals)

    def generate_matrix(self, *, use_complex_dtype: bool = False) -> np.ndarray:
        mirror_upper_triangle = self._pick_mirror_triangle_method(
            use_complex_dtype=use_complex_dtype,
        )

        lapack_heev = self._pick_lapack_heev(
            use_complex_dtype=use_complex_dtype,
        )

        eigvecs = lapack_heev(
            self.eigvecs_ensemble.generate_matrix(
                use_complex_dtype=use_complex_dtype,
            ),
            compute_v=1,
            overwrite_a=True,
        )[1]
        eigvals = self.generate_eigenvalues()

        blas_her = self._pick_blas_her(use_complex_dtype=use_complex_dtype)
        matrix = self._empty_matrix(use_complex_dtype=use_complex_dtype)
        matrix.fill(0.0)
        for mu in range(self.dimension):
            blas_her(float(eigvals[mu]), x=eigvecs[:, mu], a=matrix, overwrite_a=1)

        mirror_upper_triangle(matrix)

        return matrix

    def matrix_stream(
        self, realizs: int, *, use_complex_dtype: bool = False
    ) -> Iterator[np.ndarray]:
        mirror_upper_triangle = self._pick_mirror_triangle_method(
            use_complex_dtype=use_complex_dtype,
        )

        blas_her = self._pick_blas_her(use_complex_dtype=use_complex_dtype)
        matrix = self._empty_matrix(use_complex_dtype=use_complex_dtype)
        for eigvals, eigvecs in self.eigsys_stream(
            realizs, use_complex_dtype=use_complex_dtype
        ):
            matrix.fill(0.0)
            for mu in range(self.dimension):
                blas_her(float(eigvals[mu]), x=eigvecs[:, mu], a=matrix, overwrite_a=1)

            mirror_upper_triangle(matrix)

            yield matrix

    def eigsys_stream(
        self, realizs: int, *, use_complex_dtype: bool = False
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for _, eigvecs in self.eigvecs_ensemble.eigsys_stream(
            realizs, use_complex_dtype=use_complex_dtype
        ):
            yield self.generate_eigenvalues(), eigvecs

    def eigvals_stream(
        self, realizs: int, *, use_complex_dtype: bool = False
    ) -> Iterator[np.ndarray]:
        for _ in range(realizs):
            yield self.generate_eigenvalues()

    def spectral_pdf(self, eigvals: np.ndarray) -> np.ndarray:
        eigvals = np.asarray(eigvals)
        pdf = np.zeros_like(eigvals, dtype=np.result_type(eigvals, float))
        pdf[np.abs(eigvals) < self.spectral_radius] = 1 / 2 / self.spectral_radius
        return pdf

    def cdf(self, eigvals: np.ndarray) -> np.ndarray:
        eigvals = np.asarray(eigvals)

        in_support = np.abs(eigvals) < self.spectral_radius
        cdf = np.zeros_like(eigvals, dtype=np.result_type(eigvals, float))
        cdf[in_support] = eigvals[in_support] / (2 * self.spectral_radius) + 0.5
        cdf[eigvals >= self.spectral_radius] = 1.0
        return cdf

    def porter_thomas_distribution(
        self, widths: np.ndarray, *, num_channels: int = 1
    ) -> np.ndarray:
        return rmtpy.universal.porter_thomas_distribution(
            widths,
            dyson_index=self.eigvecs_ensemble.dyson_index,
            num_channels=num_channels,
        )

    def _empty_matrix(self, *, use_complex_dtype: bool = False) -> np.ndarray:
        if use_complex_dtype or self.eigvecs_ensemble.dyson_index != 1:
            return np.empty(
                (self.dimension, self.dimension), self.complex_dtype.type, order="F"
            )
        else:
            return np.empty(
                (self.dimension, self.dimension), self.real_dtype.type, order="F"
            )

    def _pick_blas_gemm(self, *, use_complex_dtype: bool = False) -> Callable[..., Any]:
        return self.eigvecs_ensemble._pick_blas_gemm(
            use_complex_dtype=use_complex_dtype,
        )

    def _pick_blas_her(self, *, use_complex_dtype: bool = False) -> Callable[..., Any]:
        return self.eigvecs_ensemble._pick_blas_her(
            use_complex_dtype=use_complex_dtype,
        )

    def _pick_lapack_geev(self, *, use_complex_dtype: bool = False) -> Callable[..., Any]:
        return self.eigvecs_ensemble._pick_lapack_geev(
            use_complex_dtype=use_complex_dtype,
        )

    def _pick_lapack_heev(self, *, use_complex_dtype: bool = False) -> Callable[..., Any]:
        return self.eigvecs_ensemble._pick_lapack_heev(
            use_complex_dtype=use_complex_dtype,
        )

    def _pick_mirror_triangle_method(
        self, *, use_complex_dtype: bool = False
    ) -> Callable[[np.ndarray], None]:
        if use_complex_dtype or self.eigvecs_ensemble.dyson_index != 1:
            return mirror_upper_to_lower_triangle_complex
        else:
            return mirror_upper_to_lower_triangle_real
