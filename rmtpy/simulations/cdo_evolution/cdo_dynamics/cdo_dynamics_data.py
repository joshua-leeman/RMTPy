from __future__ import annotations

import attrs
import numpy as np

import rmtpy.density
import rmtpy.validators

from ...data import Data

NUM_TIMES: int = 1000

LOGD_TIME_SUPPORT: rmtpy.density.Support = (-1.0, 1.5)


def create_times(data: CDODynamicsData) -> np.ndarray:
    times = np.zeros(data.num_times, dtype=np.float64)
    if data.num_times > 1:
        times[1:] = data.scale * rmtpy.density.array_of_floats(
            support=data.logD_time_support,
            num_pts=data.num_times - 1,
            log_base=data.dimension,
        )
    times.flags.writeable = False
    return times


def create_probabilities(data: CDODynamicsData) -> np.ndarray:
    return np.zeros((data.num_times, data.dimension), dtype=np.float64)


def create_time_series(data: CDODynamicsData) -> np.ndarray:
    return np.zeros(data.num_times, dtype=np.float64)


def create_realizs_count(_: CDODynamicsData) -> np.ndarray:
    return np.zeros(1, dtype=np.int64)


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class CDODynamicsData(Data):
    """Time-resolved statistics of the chaotic density operator."""

    dimension: int = attrs.field(
        converter=int,
        validator=attrs.validators.gt(0),
    )
    scale: float = attrs.field(
        converter=float,
        validator=attrs.validators.gt(0.0),
    )
    logD_time_support: rmtpy.density.Support = attrs.field(
        default=LOGD_TIME_SUPPORT,
        converter=tuple,
        validator=lambda _, __, support: rmtpy.validators.validate_support(support),
    )
    num_times: int = attrs.field(
        default=NUM_TIMES,
        converter=int,
        validator=attrs.validators.gt(0),
    )

    times: np.ndarray = attrs.field(
        default=attrs.Factory(create_times, takes_self=True),
        init=False,
        repr=False,
    )
    probabilities: np.ndarray = attrs.field(
        default=attrs.Factory(create_probabilities, takes_self=True),
        init=False,
        repr=False,
    )
    classical_purity: np.ndarray = attrs.field(
        default=attrs.Factory(create_time_series, takes_self=True),
        init=False,
        repr=False,
    )
    quantum_purity: np.ndarray = attrs.field(
        default=attrs.Factory(create_time_series, takes_self=True),
        init=False,
        repr=False,
    )
    entropy: np.ndarray = attrs.field(
        default=attrs.Factory(create_time_series, takes_self=True),
        init=False,
        repr=False,
    )
    kl_divergence: np.ndarray = attrs.field(
        default=attrs.Factory(create_time_series, takes_self=True),
        init=False,
        repr=False,
    )
    _realizs_count: np.ndarray = attrs.field(
        default=attrs.Factory(create_realizs_count, takes_self=True),
        init=False,
        repr=False,
    )

    @property
    def realizs(self) -> int:
        return int(self._realizs_count[0])
