from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import ClassVar

import attrs
import numpy as np

import rmtpy.conversion
import rmtpy.polynomials

from .many_body import ManyBodyEnsemble

INITIALISM: str = "WDE"

WIGNER_DYSON_ENSEMBLE_NAMES_BY_INITIALISM = {}
WIGNER_DYSON_ENSEMBLE_INITIALISMS_BY_NAME = {}


def create_spectral_weight(
    wde: WignerDysonEnsemble,
) -> Callable[[np.ndarray], np.ndarray]:
    def wigner_dyson_spectral_weight(energies: np.ndarray) -> np.ndarray:
        return rmtpy.polynomials.semicircle_weight_pdf(energies, wde.spectral_radius)

    return wigner_dyson_spectral_weight


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class WignerDysonEnsemble(ManyBodyEnsemble):
    initialism: ClassVar[str] = INITIALISM

    spectral_polynomials: Callable[[np.ndarray, int], np.ndarray] = attrs.field(
        default=rmtpy.polynomials.chebyshev_polynomials_2,
        init=False,
        repr=False,
    )
    spectral_weight: Callable[[np.ndarray], np.ndarray] = attrs.field(
        default=attrs.Factory(create_spectral_weight, takes_self=True),
        init=False,
        repr=False,
    )

    @classmethod
    def __attrs_init_subclass__(cls) -> None:
        super().__attrs_init_subclass__()
        if not inspect.isabstract(cls):
            initialism: str = rmtpy.conversion.to_registry_key(cls.initialism)
            WIGNER_DYSON_ENSEMBLE_NAMES_BY_INITIALISM[initialism] = cls.__name__.lower()
            WIGNER_DYSON_ENSEMBLE_INITIALISMS_BY_NAME[cls.__name__.lower()] = initialism
