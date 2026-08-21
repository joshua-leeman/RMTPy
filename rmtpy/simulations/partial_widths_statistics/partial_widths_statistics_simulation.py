from __future__ import annotations

from pathlib import Path
from typing import Any

import attrs

import rmtpy.conversion
from rmtpy.compounds import Compound

from ..base import Simulation
from ..statistics import REALIZATIONS_METADATA
from .outputs import PartialWidthOutputs, create_partial_width_outputs

WIDTH_INDICES: tuple[tuple[int, ...], ...] = ((0, 0), (1, 0), (1, 1), (0,), (1,))


def create_default_width_indices(
    simulation: PartialWidthsStatisticsSimulation,
) -> tuple[tuple[int, ...], ...]:
    dimension = simulation.compound.ensemble.dimension
    num_channels = simulation.compound.num_channels

    return tuple(
        width_index
        for width_index in WIDTH_INDICES
        if width_index[0] < dimension
        and (len(width_index) == 1 or width_index[1] < num_channels)
    )


def normalize_width_indices(width_indices: Any) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(index) for index in width_index) for width_index in width_indices
    )


def validate_width_indices(
    simulation: PartialWidthsStatisticsSimulation,
    _,
    width_indices: tuple[tuple[int, ...], ...],
) -> None:
    if not width_indices:
        raise ValueError("`width_indices` must contain at least one selection.")

    if len(set(width_indices)) != len(width_indices):
        raise ValueError("`width_indices` must contain unique selections.")

    dimension = simulation.compound.ensemble.dimension
    num_channels = simulation.compound.num_channels

    for width_index in width_indices:
        if len(width_index) not in (1, 2):
            raise ValueError(
                "Each width index must be `(state,)` for a total width or "
                "`(state, channel)` for a partial width."
            )

        state_index = width_index[0]
        if not 0 <= state_index < dimension:
            raise ValueError(
                f"Width state index must be in [0, {dimension}), got {state_index}."
            )

        if len(width_index) == 2:
            channel_index = width_index[1]
            if not 0 <= channel_index < num_channels:
                raise ValueError(
                    "Width channel index must be in "
                    f"[0, {num_channels}), got {channel_index}."
                )


def run_partial_widths_statistics(compound: Compound, realizs: int) -> None:
    PartialWidthsStatisticsSimulation(compound=compound, realizs=realizs).run()


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class PartialWidthsStatisticsSimulation(Simulation):
    """Monte Carlo experiment for selected partial and total decay widths."""

    compound: Compound = attrs.field(
        converter=Compound.create,
    )
    width_indices: tuple[tuple[int, ...], ...] = attrs.field(
        default=attrs.Factory(create_default_width_indices, takes_self=True),
        converter=normalize_width_indices,
        validator=validate_width_indices,
    )
    realizs: int = attrs.field(
        converter=int,
        validator=attrs.validators.gt(0),
        metadata=REALIZATIONS_METADATA,
    )

    outputs: PartialWidthOutputs = attrs.field(
        default=attrs.Factory(create_partial_width_outputs, takes_self=True),
        init=False,
        repr=False,
    )

    @property
    def to_path(self) -> Path:
        return rmtpy.conversion.to_path(
            self,
            root=Path(self.path_name) / self.compound.to_path,
        )

    def realize_monte_carlo_simulation(self) -> None:
        for partial_widths in self.compound.partial_widths_stream(realizs=self.realizs):
            self.outputs.add(partial_widths)

        self.outputs.normalize_by_average_width(self.realizs)
