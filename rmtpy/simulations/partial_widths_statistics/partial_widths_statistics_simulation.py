from __future__ import annotations

from pathlib import Path
from typing import Any

import attrs

from rmtpy.compounds import Compound
from rmtpy.conversion import RMT_CONVERTER

from ..base import Simulation
from ..statistics import REALIZATIONS_METADATA, simulation_output_path
from .outputs import PartialWidthOutputs, create_partial_width_outputs

WIDTH_INDICES: tuple[tuple[int, ...], ...] = ((0, 0), (1, 0), (1, 1), (0,), (1,))


def create_outputs(
    simulation: PartialWidthsStatisticsSimulation,
) -> PartialWidthOutputs:
    return create_partial_width_outputs(simulation)


def normalize_width_indices(width_indices: Any) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(index) for index in width_index) for width_index in width_indices
    )


def run_partial_widths_statistics(compound: Compound, realizs: int) -> None:
    PartialWidthsStatisticsSimulation(compound=compound, realizs=realizs).run()


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class PartialWidthsStatisticsSimulation(Simulation):
    compound: Compound = attrs.field(
        converter=Compound.create,
    )
    width_indices: tuple[tuple[int, ...], ...] = attrs.field(
        default=WIDTH_INDICES,
        converter=normalize_width_indices,
    )
    realizs: int = attrs.field(
        converter=int,
        validator=attrs.validators.gt(0),
        metadata=REALIZATIONS_METADATA,
    )

    outputs: PartialWidthOutputs = attrs.field(
        default=attrs.Factory(create_outputs, takes_self=True),
        init=False,
        repr=False,
    )

    @property
    def to_path(self) -> Path:
        return simulation_output_path(self, Path(self.path_name) / self.compound.to_path)

    def populate_metadata(self) -> None:
        super().populate_metadata()

        self.metadata["args"]["realizs"] = self.realizs
        self.metadata["args"]["compound"] = RMT_CONVERTER.unstructure(self.compound)
        self.metadata["args"]["width_indices"] = self.width_indices

    def realize_monte_carlo_simulation(self) -> None:
        for partial_widths in self.compound.partial_widths_stream(realizs=self.realizs):
            self.outputs.add(partial_widths)

        self.outputs.normalize_by_average_width(self.realizs)
