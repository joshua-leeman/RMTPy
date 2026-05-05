from __future__ import annotations

import numpy as np
from attrs import field, frozen

from ._many_body import ManyBodyEnsemble
from ..utils.polynomials import chebyshev_polynomials_2

WIGNER_DYSON_ENSEMBLE_FLAGS_TO_NAMES = {
    "bdgc": "bogoliubovdegennescensemble",
    "bdgd": "bogoliubovdegennesdensemble",
    "goe": "gaussianorthogonalensemble",
    "gue": "gaussianunitaryensemble",
    "gse": "gaussiansymplecticensemble",
}
WIGNER_DYSON_ENSEMBLE_NAMES_TO_FLAGS = {
    v: k for k, v in WIGNER_DYSON_ENSEMBLE_FLAGS_TO_NAMES.items()
}


@frozen(kw_only=True)
class WignerDysonEnsemble(ManyBodyEnsemble):
    _nickname: str = field(init=False, default="WDE", repr=False)

    _spectral_coeffs_max_order: int = field(init=False, default=10, repr=False)
    _average_spectral_coeffs: np.ndarray | None = field(
        init=False, default=None, repr=False
    )

    def average_spectral_coeffs(self, max_order: int = 10) -> np.ndarray:
        cached_max_order: int = self._spectral_coeffs_max_order
        cached_coeffs: np.ndarray | None = self._average_spectral_coeffs

        if max_order < 0 or not isinstance(max_order, int):
            raise ValueError("max_order must be a non-negative integer")

        if max_order > cached_max_order:
            object.__setattr__(self, "_spectral_coeffs_max_order", max_order)
            object.__setattr__(self, "_average_spectral_coeffs", None)

        if cached_coeffs is None:
            object.__setattr__(
                self,
                "_average_spectral_coeffs",
                self._create_average_spectral_coeffs(max_order),
            )

        elif max_order < cached_max_order:
            return cached_coeffs[: max_order + 1]

        return self._average_spectral_coeffs

    def semicircle_pdf(self, energies: int | float | np.ndarray) -> np.ndarray:
        real_dtype: type[np.floating] = self.real_dtype.type
        energy_0: float = self.ground_state_energy

        if isinstance(energies, (int, float)):
            energies: np.ndarray = np.array([energies], dtype=real_dtype)

        x: np.ndarray = energies / energy_0
        mask: np.ndarray = np.abs(x) < 1.0
        pdf: np.ndarray = np.zeros_like(x, real_dtype)
        pdf[mask] = np.sqrt(1 - x[mask] * x[mask])
        pdf[mask] *= 2 / np.pi / energy_0
        return pdf

    def average_spectral_pdf(
        self, energies: int | float | np.ndarray, max_order: int = 10
    ) -> np.ndarray:
        real_dtype: type[np.floating] = self.real_dtype.type
        average_coeffs: np.ndarray = self.average_spectral_coeffs(max_order)

        if isinstance(energies, (int, float)):
            energies: np.ndarray = np.array([energies], dtype=real_dtype)

        x: np.ndarray = energies / self.ground_state_energy

        polynomials: np.ndarray = np.empty((max_order + 1, energies.size))
        chebyshev_polynomials_2(max_order, x, out=polynomials)
        polynomials *= average_coeffs[:, None]

        weight_function: np.ndarray = self.semicircle_pdf(energies)
        average_pdf: np.ndarray = weight_function * np.sum(polynomials, axis=0)
        return average_pdf

    def variate_spectral_pdf(
        self,
        eigvals: int | float | np.ndarray,
        energies: int | float | np.ndarray | None = None,
        max_order: int = 10,
    ) -> np.ndarray:
        real_dtype: type[np.floating] = self.real_dtype.type
        if isinstance(eigvals, (int, float)):
            eigvals: np.ndarray = np.array([eigvals], dtype=real_dtype)
        if isinstance(energies, (int, float)):
            energies: np.ndarray = np.array([energies], dtype=real_dtype)

        polynomials: np.ndarray = np.empty((max_order + 1, self.dimension))
        x: np.ndarray = eigvals / self.ground_state_energy
        chebyshev_polynomials_2(max_order, x, out=polynomials)
        spectral_coeffs: np.ndarray = np.mean(polynomials, axis=1)

        if energies is not None or not np.allclose(eigvals, energies):
            polynomials: np.ndarray = np.empty((max_order + 1, energies.size))
            x: np.ndarray = energies / self.ground_state_energy
            chebyshev_polynomials_2(max_order, x, out=polynomials)

        polynomials *= spectral_coeffs[:, None]

        weight_function: np.ndarray = self.semicircle_pdf(energies)
        variate_pdf: np.ndarray = weight_function * np.sum(polynomials, axis=0)
        return variate_pdf

    def _create_average_spectral_coeffs(self, max_order: int = 10) -> np.ndarray:
        total_counts_per_dimension: int = 2**13 // self.dimension
        realizs: int = 10 * max(total_counts_per_dimension, 1)

        polynomials: np.ndarray = np.empty((max_order + 1, self.dimension))
        average_coeffs: np.ndarray = np.zeros(max_order + 1)

        for tmp_eigvals in self.eigvals_stream(realizs):
            x: np.ndarray = tmp_eigvals / self.ground_state_energy
            chebyshev_polynomials_2(max_order, x, out=polynomials)
            average_coeffs += np.mean(polynomials, axis=1)

        average_coeffs /= realizs
        return average_coeffs
