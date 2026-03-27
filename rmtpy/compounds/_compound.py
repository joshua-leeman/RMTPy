# rmtpy/compounds/_compound.py

# Postponed evaluation of annotations
from __future__ import annotations

# Standard library imports
import math

# Local application imports
from ..ensembles import ManyBodyEnsemble
from collections.abc import Iterator

# Third-party imports
import numpy as np
from attrs import field, frozen
from attrs.validators import instance_of


# ---------------------------
# Many-body Compound Ensemble
# ---------------------------
@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class Compound:

    # Underlying RMT ensemble
    ensemble: ManyBodyEnsemble = field(validator=instance_of(ManyBodyEnsemble))

    # Number of asymptotically-free fermions
    fermions: int = field(default=1, converter=int)

    # Number of open channels
    channels: int = field(init=False)

    # Coupling strengths
    couplings: float = field()

    @fermions.validator
    def __fermions_validator(self, _, value: int) -> None:
        """Ensure number of fermions is a positive integer."""

        # Alias number of complex fermions in ensemble
        Nc = self.ensemble.N // 2

        # =================================================

        # Ensure number of fermions is a positive integer less than or equal to Nc
        if value <= 0 or value > Nc:
            raise ValueError(
                f"Number of fermions must be a positive integer less than or equal to {Nc}, got {value}."
            )

    @channels.default
    def __channels_default(self) -> int:
        """Set number of open channels equal to number of combinations of complex fermions from the ensemble."""

        # Alias number of asymptotically-free fermions
        k = self.fermions

        # Alias number of complex fermions in ensemble
        Nc = self.ensemble.N // 2

        # =================================================

        # Calculate and return number of combinations
        return math.comb(Nc, k)
