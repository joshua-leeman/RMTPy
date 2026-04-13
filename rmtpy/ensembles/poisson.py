from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
from attrs import field, frozen
from scipy.linalg import eigh

from ._many_body import ManyBodyEnsemble
from ._wigner_dyson import WIGNER_DYSON_ENSEMBLE_FLAGS, WignerDysonEnsemble
from ..utils import rmtpy_converter


@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class PoissonEnsemble(ManyBodyEnsemble):
    eigvecs_flag: str = field(default="GaussianUnitaryEnsemble")

    @eigvecs_flag.validator
    def _eigvecs_flag_validator(self, _: field, value: str) -> None:
        if value not in WIGNER_DYSON_ENSEMBLE_FLAGS:
            raise ValueError(
                f"eigvecs_flag must indicate a Wigner-Dyson ensemble, got {value}"
            )

    eigvecs_ensemble: WignerDysonEnsemble = field(init=False, repr=False)
    dyson_index: int = field(init=False, default=0, repr=False)
    std_dev: float = field(init=False, repr=False)

    @std_dev.default
    def _default_std_dev(self) -> float:
        return 2 * self.ground_state_energy

    _nickname: str = field(init=False, default="Poisson", repr=False)

    def __attrs_post_init__(self) -> None:
        ensemble_dict: dict[str, Any] = rmtpy_converter.unstructure(self)
        ensemble_dict["name"] = self.eigvecs_flag
        ensemble: WignerDysonEnsemble = rmtpy_converter.structure(
            ensemble_dict, WignerDysonEnsemble
        )
        object.__setattr__(self, "eigvecs_ensemble", ensemble)

    def generate_matrix(self) -> np.ndarray:
        raise NotImplementedError(
            "Matrix generation is not implemented for the Poisson ensemble."
        )

    def matrix_stream(
        self, realizs: int, use_complex_dtype: bool = True
    ) -> Iterator[np.ndarray]:
        raise NotImplementedError(
            "Matrix stream is not implemented for the Poisson ensemble."
        )

    def eigsys_stream(self, realizs: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        real_dtype: type[np.floating] = self.real_dtype.type
        rng: np.random.Generator = self.rng
        dimension: int = self.dimension
        std_dev: float = self.std_dev
        eigvecs_ensemble: ManyBodyEnsemble = self.eigvecs_ensemble

        for _ in range(realizs):
            eigvals: np.ndarray = rng.random(dimension, real_dtype)
            eigvals -= 0.5
            eigvals *= std_dev
            eigvals.sort()
            eigvecs_ensemble.generate_matrix(out=U)
            _, U = eigh(U, overwrite_a=True, check_finite=False)
            yield eigvals, U

    def eigvals_stream(self, realizs: int) -> Iterator[np.ndarray]:
        real_dtype: type[np.floating] = self.real_dtype.type
        rng: np.random.Generator = self.rng
        dimension: int = self.dimension
        std_dev: float = self.std_dev

        for _ in range(realizs):
            eigvals: np.ndarray = rng.random(dimension, real_dtype)
            eigvals -= 0.5
            eigvals *= std_dev
            eigvals.sort()
            yield eigvals

    def spectral_pdf(
        self,
        eigvals: int | float | np.ndarray,
        _realizs: int = 100,
        _factor: float = 1.2,
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
