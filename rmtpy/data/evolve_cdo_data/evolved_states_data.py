# rmtpy/data/evolve_cdo_data/evolved_states_data.py

# Postponed evaluation of annotations
from __future__ import annotations

# Third-party imports
import numpy as np
from attrs import frozen, field
from attrs.validators import gt

# Local application imports
from .._data import Data


# -------------------------
# Evolved States Data Class
# -------------------------
@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class EvolvedStatesData(Data):

    # Number of realizations
    realizs: int = field(converter=int, validator=gt(0))

    # Number of times for evolved states
    num_times: int = field(converter=int, validator=gt(0))

    # Hilbert space dimension
    dim: int = field(converter=int, validator=gt(0))

    # Ensemble data type
    dtype: np.dtype = field(converter=lambda dt: np.dtype(dt))

    # Evolved states numpy array
    states: np.ndarray = field(repr=False)

    @states.default
    def __states_default(self) -> np.ndarray:
        """Initialize the evolved states array."""

        # Return empty array with shape R x T x D
        return np.empty(
            (self.realizs, self.num_times, self.dim), dtype=self.dtype, order="F"
        )
