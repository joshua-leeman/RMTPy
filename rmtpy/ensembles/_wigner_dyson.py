from __future__ import annotations

import numpy as np
from attrs import field, frozen

from ._many_body import ManyBodyEnsemble


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

        x: np.ndarray = eigvals / energy_0
        mask: np.ndarray = np.abs(x) < 1.0
        pdf: np.ndarray = np.zeros_like(x, real_dtype)
        pdf[mask] = np.sqrt(1 - x[mask] * x[mask])
        pdf[mask] *= 2 / np.pi / energy_0
        return pdf

    def spectral_cdf(self, eigvals: int | float | np.ndarray) -> np.ndarray:
        energy_0: float = self.ground_state_energy

        if isinstance(eigvals, (int, float)):
            eigvals: np.ndarray = np.array([eigvals], dtype=self.real_dtype.type)

        x: np.ndarray = np.clip(eigvals / energy_0, -1.0, 1.0)
        cdf: np.ndarray = np.sqrt(1 - x * x)
        cdf *= x
        cdf += np.arcsin(x)
        cdf /= np.pi
        cdf += 0.5
        return cdf
