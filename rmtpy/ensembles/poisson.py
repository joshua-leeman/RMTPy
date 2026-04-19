from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
from attrs import field, frozen
from numba import njit

from ._many_body import ManyBodyEnsemble
from ._wigner_dyson import (
    WIGNER_DYSON_ENSEMBLE_FLAGS_TO_NAMES,
    WIGNER_DYSON_ENSEMBLE_NAMES_TO_FLAGS,
    WignerDysonEnsemble,
)
from ..utils import rmtpy_converter


@njit(cache=True, fastmath=True)
def _create_poisson_matrix(
    matrix: np.ndarray, eigvals: np.ndarray, eigvecs: np.ndarray
) -> np.ndarray:
    size: int = matrix.shape[0]
    matrix.fill(0.0)
    for i in range(size):
        for j in range(size):
            for k in range(size):
                matrix[i, j] += eigvals[k] * eigvecs[i, k] * eigvecs[j, k]


@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class PoissonEnsemble(ManyBodyEnsemble):
    eigvecs_flag: str = field(default="gue", converter=str.lower)

    @eigvecs_flag.validator
    def _eigvecs_flag_validator(self, _: field, value: str) -> None:
        if value in WIGNER_DYSON_ENSEMBLE_NAMES_TO_FLAGS:
            return WIGNER_DYSON_ENSEMBLE_NAMES_TO_FLAGS[value]
        elif value in WIGNER_DYSON_ENSEMBLE_FLAGS_TO_NAMES:
            return value
        else:
            raise ValueError(
                f"Invalid eigvecs_flag: {value}. Must be one of {list(WIGNER_DYSON_ENSEMBLE_NAMES_TO_FLAGS.keys())} or {list(WIGNER_DYSON_ENSEMBLE_FLAGS_TO_NAMES.keys())}."
            )

    eigvecs_ensemble: WignerDysonEnsemble | None = field(
        init=False, default=None, repr=False
    )
    dyson_index: int = field(init=False, default=0, repr=False)
    std_dev: float = field(init=False, repr=False)

    @std_dev.default
    def _default_std_dev(self) -> float:
        return 2 * self.ground_state_energy

    _nickname: str = field(init=False, default="Poisson", repr=False)

    def __attrs_post_init__(self) -> None:
        flag: str = self.eigvecs_flag
        eigvecs_ensemble_dict: dict[str, Any] = rmtpy_converter.unstructure(self)
        eigvecs_ensemble_dict["name"] = WIGNER_DYSON_ENSEMBLE_FLAGS_TO_NAMES[flag]
        eigvecs_ensemble: WignerDysonEnsemble = rmtpy_converter.structure(
            eigvecs_ensemble_dict, WignerDysonEnsemble
        )
        object.__setattr__(self, "eigvecs_ensemble", eigvecs_ensemble)

    @property
    def _path_name(self) -> str:
        return super()._path_name + f"_{self.eigvecs_ensemble._nickname.lower()}"

    def generate_matrix(self, use_complex_dtype: bool = False) -> np.ndarray:
        complex_dtype: type[np.complexfloating] = self.complex_dtype.type
        real_dtype: type[np.floating] = self.real_dtype.type
        rng: np.random.Generator = self.rng
        dim: int = self.dimension
        std_dev: float = self.std_dev
        eigvecs_ensemble: WignerDysonEnsemble = self.eigvecs_ensemble
        eigvecs_ensemble_dyson_index: int = self.eigvecs_ensemble.dyson_index

        lapack_heev: type = eigvecs_ensemble._pick_lapack_heev(use_complex_dtype)

        if use_complex_dtype or eigvecs_ensemble_dyson_index != 1:
            matrix: np.ndarray = np.empty((dim, dim), complex_dtype, order="F")
        else:
            matrix: np.ndarray = np.empty((dim, dim), real_dtype, order="F")

        eigvals: np.ndarray = rng.random(dim, real_dtype)
        eigvals -= 0.5
        eigvals *= std_dev

        eigvecs: np.ndarray = lapack_heev(
            eigvecs_ensemble.generate_matrix(use_complex_dtype),
            compute_v=1,
            overwrite_a=True,
        )[1]

        _create_poisson_matrix(matrix, eigvals, eigvecs)
        return matrix

    def matrix_stream(
        self, realizs: int, use_complex_dtype: bool = False
    ) -> Iterator[np.ndarray]:
        complex_dtype: type[np.complexfloating] = self.complex_dtype.type
        real_dtype: type[np.floating] = self.real_dtype.type
        eigvecs_ensemble_dyson_index: int = self.eigvecs_ensemble.dyson_index
        dim: int = self.dimension

        if use_complex_dtype or eigvecs_ensemble_dyson_index != 1:
            matrix: np.ndarray = np.empty((dim, dim), complex_dtype, order="F")
        else:
            matrix: np.ndarray = np.empty((dim, dim), real_dtype, order="F")

        for eigvals, eigvecs in self.eigsys_stream(realizs, use_complex_dtype):
            _create_poisson_matrix(matrix, eigvals, eigvecs)
            yield matrix

    def eigsys_stream(
        self, realizs: int, use_complex_dtype: bool = False
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        real_dtype: type[np.floating] = self.real_dtype.type
        rng: np.random.Generator = self.rng
        dimension: int = self.dimension
        std_dev: float = self.std_dev
        eigvecs_ensemble: ManyBodyEnsemble = self.eigvecs_ensemble

        for _, eigvecs in eigvecs_ensemble.eigsys_stream(realizs, use_complex_dtype):
            eigvals: np.ndarray = rng.random(dimension, real_dtype)
            eigvals -= 0.5
            eigvals *= std_dev
            yield eigvals, eigvecs

    def eigvals_stream(self, realizs: int) -> Iterator[np.ndarray]:
        real_dtype: type[np.floating] = self.real_dtype.type
        rng: np.random.Generator = self.rng
        dimension: int = self.dimension
        std_dev: float = self.std_dev

        for _ in range(realizs):
            eigvals: np.ndarray = rng.random(dimension, real_dtype)
            eigvals -= 0.5
            eigvals *= std_dev
            yield eigvals

    def spectral_pdf(
        self,
        eigvals: int | float | np.ndarray,
        _num_bins: int = 200,
        _factor: float = 1.2,
        _sigma: float = 2.0,
    ) -> np.ndarray:
        real_dtype: type[np.floating] = self.real_dtype.type
        energy_0: float = self.ground_state_energy

        if isinstance(eigvals, (int, float)):
            eigvals: np.ndarray = np.array([eigvals], dtype=real_dtype)

        pdf: np.ndarray = np.zeros_like(eigvals, real_dtype)
        pdf[np.abs(eigvals) < energy_0] = 1 / 2 / energy_0
        return pdf

    def cdf(self, eigvals: int | float | np.ndarray) -> np.ndarray:
        real_dtype: type[np.floating] = self.real_dtype.type
        energy_0: float = self.ground_state_energy

        if isinstance(eigvals, (int, float)):
            eigvals: np.ndarray = np.array([eigvals], dtype=real_dtype)

        cdf: np.ndarray = np.zeros_like(eigvals, real_dtype)
        mask: np.ndarray = np.abs(eigvals) < energy_0
        cdf[mask] = eigvals[mask] / (2 * energy_0) + 0.5
        cdf[eigvals > energy_0] = 1.0
        return cdf

    def _pick_blas_copy(self, use_complex_dtype: bool) -> type:
        return self.eigvecs_ensemble._pick_blas_copy(use_complex_dtype)

    def _pick_blas_gemm(self, use_complex_dtype: bool) -> type:
        return self.eigvecs_ensemble._pick_blas_gemm(use_complex_dtype)

    def _pick_lapack_geev(self, use_complex_dtype: bool) -> type:
        return self.eigvecs_ensemble._pick_lapack_geev(use_complex_dtype)

    def _pick_lapack_heev(self, use_complex_dtype: bool) -> type:
        return self.eigvecs_ensemble._pick_lapack_heev(use_complex_dtype)
