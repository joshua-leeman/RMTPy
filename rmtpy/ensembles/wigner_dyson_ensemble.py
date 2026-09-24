import inspect
from abc import ABC
from typing import ClassVar, override

import attrs
import numpy as np

from ..conversion import to_key_of_registry
from ..polynomials import (
    FloatFunction,
    OrthogonalPolynomials,
    chebyshev_polynomial_2_weight,
    chebyshev_polynomials_2,
)
from .many_body_ensemble import ManyBodyEnsemble

INITIALISM: str = "WDE"

WIGNER_DYSON_ENSEMBLE_INITIALISMS_BY_NAME: dict[str, str] = {}
WIGNER_DYSON_ENSEMBLE_NAMES_BY_INITIALISM: dict[str, str] = {}


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class WignerDysonEnsemble(ManyBodyEnsemble, ABC):
    initialism: ClassVar[str] = INITIALISM

    @classmethod
    @override
    def __attrs_init_subclass__(cls) -> None:
        super().__attrs_init_subclass__()

        if not inspect.isabstract(cls):
            initialism = to_key_of_registry(cls.initialism)

            WIGNER_DYSON_ENSEMBLE_NAMES_BY_INITIALISM[initialism] = cls.__name__.lower()
            WIGNER_DYSON_ENSEMBLE_INITIALISMS_BY_NAME[cls.__name__.lower()] = initialism

    @override
    def assign_spectral_polynomials(self) -> OrthogonalPolynomials:
        def wigner_dyson_spectral_polynomials(
            x: np.ndarray[tuple[int], np.dtype[np.floating]],
            *,
            degree: int,
        ) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
            return chebyshev_polynomials_2(x, degree=degree)

        return wigner_dyson_spectral_polynomials

    @override
    def assign_spectral_weight(self) -> FloatFunction:
        def wigner_semicircle_distribution(
            energies: np.ndarray[tuple[int], np.dtype[np.floating]],
        ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
            return chebyshev_polynomial_2_weight(
                energies,
                support_radius=self.spectral_radius,
            )

        return wigner_semicircle_distribution
