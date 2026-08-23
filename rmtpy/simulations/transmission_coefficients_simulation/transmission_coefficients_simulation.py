from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import attrs
import numpy as np

import rmtpy.conversion
from rmtpy.compounds import Compound

from ..base import Simulation
from ..observable import Observable
from ..statistics import REALIZATIONS_METADATA
from .outputs import (
    TransmissionCoefficientsOutputs,
    create_transmission_coefficients_outputs,
)

NUM_ENERGY_POINTS: int = 100


def normalize_channel_indices(channel_indices: Any) -> tuple[int, ...]:
    if np.isscalar(channel_indices):
        channel_indices_array = np.asarray([channel_indices])
    else:
        try:
            channel_indices_array = np.asarray(tuple(channel_indices))
        except TypeError:
            channel_indices_array = np.asarray([channel_indices])

    if channel_indices_array.ndim == 0:
        channel_indices_array = channel_indices_array.reshape(1)
    if channel_indices_array.ndim != 1:
        raise ValueError("`channel_indices` must be a one-dimensional array-like.")

    try:
        normalized_indices = tuple(int(index) for index in channel_indices_array)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(
            "`channel_indices` must contain integer channel indices."
        ) from exc

    if any(
        index != original
        for index, original in zip(normalized_indices, channel_indices_array, strict=True)
    ):
        raise ValueError("`channel_indices` must contain integer channel indices.")

    return normalized_indices


def validate_channel_indices(
    simulation: TransmissionCoefficientsSimulation,
    _,
    channel_indices: tuple[int, ...],
) -> None:
    if not channel_indices:
        raise ValueError("`channel_indices` must contain at least one channel index.")
    if len(set(channel_indices)) != len(channel_indices):
        raise ValueError("`channel_indices` must contain unique channel indices.")

    for channel_index in channel_indices:
        if not 0 <= channel_index < simulation.compound.num_channels:
            raise ValueError(
                "Channel index must be in "
                f"[0, {simulation.compound.num_channels}), got {channel_index}."
            )


def create_energy_grid(simulation: TransmissionCoefficientsSimulation) -> np.ndarray:
    spectral_density = simulation.compound.ensemble.spectral_density
    energies = np.linspace(
        *spectral_density.plot_range,
        NUM_ENERGY_POINTS,
        dtype=np.float64,
    )
    energies.flags.writeable = False
    return energies


def run_transmission_coefficients_simulation(
    compound: Compound,
    *,
    realizs: int,
    channel_indices: Iterable[int] = (0,),
) -> None:
    kwargs: dict[str, Any] = {
        "compound": compound,
        "realizs": realizs,
    }
    if channel_indices is not None:
        kwargs["channel_indices"] = channel_indices
    TransmissionCoefficientsSimulation(**kwargs).run()


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class TransmissionCoefficientsSimulation(Simulation):
    """Transmission curves and the all-channel Weisskopf width estimate."""

    compound: Compound = attrs.field(
        converter=Compound.create,
    )
    realizs: int = attrs.field(
        converter=int,
        validator=attrs.validators.gt(0),
        metadata=REALIZATIONS_METADATA,
    )
    channel_indices: tuple[int, ...] = attrs.field(
        default=(0,),
        converter=normalize_channel_indices,
        validator=validate_channel_indices,
    )
    energies: np.ndarray = attrs.field(
        default=attrs.Factory(create_energy_grid, takes_self=True),
        init=False,
        repr=False,
    )

    outputs: TransmissionCoefficientsOutputs = attrs.field(
        default=attrs.Factory(create_transmission_coefficients_outputs, takes_self=True),
        init=False,
        repr=False,
    )

    @property
    def to_path(self) -> Path:
        return rmtpy.conversion.to_path(
            self,
            root=Path(self.path_name) / self.compound.to_path,
        )

    def channel_path(self, channel_index: int) -> Path:
        return Path(f"channel_{channel_index}")

    def observable_output_path(self, observable: Observable) -> Path:
        channel_index = observable.metadata.get("channel_index")
        if channel_index is None:
            return Path()
        return self.channel_path(channel_index)

    def realize_monte_carlo_simulation(self) -> None:
        for scattering_matrices, _ in self.compound.scattering_matrix_stream(
            energies=self.energies,
            realizs=self.realizs,
        ):
            self.outputs.add_scattering_matrices(scattering_matrices)
