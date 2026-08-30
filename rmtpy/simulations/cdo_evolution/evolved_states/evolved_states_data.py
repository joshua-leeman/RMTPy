from __future__ import annotations

import attrs
import numpy as np

from ...data import Data


def create_states(data: EvolvedStatesData) -> np.ndarray:
    return np.zeros(
        (data.realizs, data.num_times, data.dimension),
        dtype=data.dtype,
    )


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class EvolvedStatesData(Data):
    """Optional persisted pure states from each Monte Carlo realization."""

    realizs: int = attrs.field(
        converter=int,
        validator=attrs.validators.gt(0),
    )
    num_times: int = attrs.field(
        converter=int,
        validator=attrs.validators.gt(0),
    )
    dimension: int = attrs.field(
        converter=int,
        validator=attrs.validators.gt(0),
    )
    dtype: np.dtype = attrs.field(converter=np.dtype)

    states: np.ndarray = attrs.field(
        default=attrs.Factory(create_states, takes_self=True),
        init=False,
        repr=False,
    )
