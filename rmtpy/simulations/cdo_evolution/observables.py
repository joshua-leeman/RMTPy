from __future__ import annotations

import numpy as np

from ..observable import Observable
from ..statistics import create_observable
from .cdo_dynamics import (
    CDODynamicsData,
    CDOInformationPlot,
    CDOProbabilitiesPlot,
    CDOPuritiesPlot,
)
from .evolved_states import EvolvedStatesData


def create_cdo_dynamics_observable(
    *,
    dimension: int,
    scale: float,
    logD_time_support: tuple[float, float],
    num_times: int,
) -> Observable[CDODynamicsData]:
    return create_observable(
        data=CDODynamicsData(
            file_name="cdo_dynamics",
            dimension=dimension,
            scale=scale,
            logD_time_support=logD_time_support,
            num_times=num_times,
        ),
        plot_cls=CDOProbabilitiesPlot,
        additional_plot_classes=(CDOPuritiesPlot, CDOInformationPlot),
    )


def create_evolved_states_observable(
    *,
    realizs: int,
    num_times: int,
    dimension: int,
    dtype: np.dtype,
) -> Observable[EvolvedStatesData]:
    return create_observable(
        data=EvolvedStatesData(
            file_name="evolved_states",
            realizs=realizs,
            num_times=num_times,
            dimension=dimension,
            dtype=dtype,
        ),
    )
