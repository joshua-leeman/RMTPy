import inspect
from abc import ABC
from typing import ClassVar, override

import attrs
import numpy as np
from numpy.typing import NDArray

import rmtpy.conversion
from rmtpy.polynomials import (
    Float64Function,
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
            initialism = rmtpy.conversion.to_registry_key(cls.initialism)

            WIGNER_DYSON_ENSEMBLE_NAMES_BY_INITIALISM[initialism] = cls.__name__.lower()
            WIGNER_DYSON_ENSEMBLE_INITIALISMS_BY_NAME[cls.__name__.lower()] = initialism

    @override
    def _create_spectral_polynomials(self) -> OrthogonalPolynomials:
        def spectral_polynomials(
            x: NDArray[np.float64],
            *,
            degree: int,
        ) -> NDArray[np.float64]:
            return chebyshev_polynomials_2(x, degree=degree)

        return spectral_polynomials

    @override
    def _create_spectral_weight(self) -> Float64Function:
        def wigner_semicircle_distribution(
            energies: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            return chebyshev_polynomial_2_weight(
                energies, support_radius=self.spectral_radius
            )

        return wigner_semicircle_distribution
