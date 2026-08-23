from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import attrs
import numpy as np

from ..observable import Observable
from .observables import (
    create_transmission_coefficients_observable,
    create_weisskopf_estimate_observable,
)
from .transmission_coefficients import TransmissionCoefficientsData
from .weisskopf_estimate import WeisskopfEstimateData

if TYPE_CHECKING:
    from .transmission_coefficients_simulation import TransmissionCoefficientsSimulation


def create_transmission_coefficients_outputs(
    simulation: TransmissionCoefficientsSimulation,
) -> TransmissionCoefficientsOutputs:
    return TransmissionCoefficientsOutputs(
        by_channel=tuple(
            create_transmission_coefficients_observable(
                simulation,
                channel_index=channel_index,
            )
            for channel_index in simulation.channel_indices
        ),
        weisskopf_estimate=create_weisskopf_estimate_observable(simulation),
    )


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class TransmissionCoefficientsOutputs:
    """Per-channel transmission curves and the all-channel Weisskopf estimate."""

    by_channel: tuple[Observable[TransmissionCoefficientsData], ...]
    weisskopf_estimate: Observable[WeisskopfEstimateData]

    def iter_observables(self) -> Iterator[Observable]:
        yield from self.by_channel
        yield self.weisskopf_estimate

    def add_scattering_matrices(self, scattering_matrices: np.ndarray) -> None:
        scattering_matrices = np.asarray(scattering_matrices)
        num_energies = len(self.weisskopf_estimate.data.energies)
        num_channels = self.weisskopf_estimate.data.num_channels
        expected_shape = (num_energies, num_channels, num_channels)
        if scattering_matrices.shape != expected_shape:
            raise ValueError(
                f"Scattering matrices must have shape {expected_shape}, got "
                f"{scattering_matrices.shape}."
            )

        scattering_diagonal = np.diagonal(
            scattering_matrices,
            axis1=1,
            axis2=2,
        )
        for observable in self.by_channel:
            channel_index = observable.data.channel_index
            if channel_index >= scattering_diagonal.shape[1]:
                raise ValueError(
                    "Scattering matrices contain only "
                    f"{scattering_diagonal.shape[1]} channels, but channel "
                    f"{channel_index} was requested."
                )
            observable.data.add_scattering_diagonal(
                scattering_diagonal[:, channel_index]
            )

        self.weisskopf_estimate.data.add_scattering_diagonal(scattering_diagonal)
