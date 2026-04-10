from __future__ import annotations

import numpy as np
from attrs import frozen, field
from attrs.validators import gt

from ..._data import Data


@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class EvolvedStatesData(Data):

    realizs: int = field(converter=int, validator=gt(0))

    num_times: int = field(converter=int, validator=gt(0))

    dim: int = field(converter=int, validator=gt(0))

    dtype: np.dtype = field(converter=lambda dt: np.dtype(dt))

    states: np.ndarray = field(repr=False)

    @states.default
    def __states_default(self) -> np.ndarray:
        """Initialize the evolved states array."""

        return np.empty(
            (self.realizs, self.num_times, self.dim), dtype=self.dtype, order="F"
        )
