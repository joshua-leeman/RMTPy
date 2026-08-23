from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..observable import Observable
from ..statistics import create_observable
from .transmission_coefficients import (
    TransmissionCoefficientsData,
    TransmissionCoefficientsPlot,
)
from .weisskopf_estimate import WeisskopfEstimateData, WeisskopfEstimatePlot

if TYPE_CHECKING:
    from .transmission_coefficients_simulation import TransmissionCoefficientsSimulation


def finalize_transmission_coefficients(data: TransmissionCoefficientsData) -> None:
    data.compute_transmission_coefficients()


def finalize_weisskopf_estimate(data: WeisskopfEstimateData) -> None:
    data.compute_weisskopf_estimate()


def create_transmission_coefficients_observable(
    simulation: TransmissionCoefficientsSimulation,
    *,
    channel_index: int,
) -> Observable[TransmissionCoefficientsData]:
    return create_observable(
        data=TransmissionCoefficientsData(
            file_name="transmission_coefficients",
            energies=simulation.energies,
            channel_index=channel_index,
        ),
        plot_cls=TransmissionCoefficientsPlot,
        metadata={"channel_index": channel_index},
        finalize=finalize_transmission_coefficients,
    )


def create_weisskopf_estimate_observable(
    simulation: TransmissionCoefficientsSimulation,
) -> Observable[WeisskopfEstimateData]:
    ensemble = simulation.compound.ensemble
    weight_density = ensemble.spectral_density.weight_pdf(simulation.energies)
    mean_level_spacings = np.full_like(weight_density, np.nan, dtype=np.float64)
    positive_density = np.isfinite(weight_density) & (weight_density > 0.0)
    np.divide(
        1.0,
        ensemble.dimension * weight_density,
        out=mean_level_spacings,
        where=positive_density,
    )

    return create_observable(
        data=WeisskopfEstimateData(
            file_name="weisskopf_estimate",
            energies=simulation.energies,
            mean_level_spacings=mean_level_spacings,
            num_channels=simulation.compound.num_channels,
        ),
        plot_cls=WeisskopfEstimatePlot,
        metadata={"num_channels": simulation.compound.num_channels},
        finalize=finalize_weisskopf_estimate,
    )
