from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import attrs
import numpy as np

from rmtpy.compounds import Compound
from rmtpy.conversion import RMT_CONVERTER

from ..base import Simulation
from ..histogram import Histogram
from ..observable import Observable
from .observables import create_width_histograms

WIDTH_INDICES_DEFAULT: tuple[tuple[int, ...], ...] = (
    (0, 0),
    (1, 0),
    (1, 1),
    (0,),
    (1,),
)
REALIZATIONS_METADATA: dict[str, str] = {
    "dir_name": "realizs",
    "latex_name": "R",
}


def normalize_width_indices(width_indices: Any) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(index) for index in width_index) for width_index in width_indices
    )


def compute_width_value(
    partial_widths: np.ndarray,
    width_index: tuple[int, ...],
) -> float:
    if len(width_index) == 2:
        return float(partial_widths[width_index[0]][width_index[1]])
    if len(width_index) == 1:
        return float(np.sum(partial_widths[width_index[0]]))
    raise ValueError("Invalid width index in histogram metadata.")


def run_partial_widths_statistics(compound: Compound, realizs: int) -> None:
    PartialWidthsStatisticsSimulation(compound=compound, realizs=realizs).run()


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class PartialWidthsStatisticsSimulation(Simulation):
    compound: Compound = attrs.field(
        converter=Compound.create,
    )
    width_indices: tuple[tuple[int, ...], ...] = attrs.field(
        default=WIDTH_INDICES_DEFAULT,
        converter=normalize_width_indices,
    )
    realizs: int = attrs.field(
        converter=int,
        validator=attrs.validators.gt(0),
        metadata=REALIZATIONS_METADATA,
    )

    width_histograms: list[Observable] = attrs.field(
        default=attrs.Factory(create_width_histograms, takes_self=True),
        init=False,
        repr=False,
    )

    @property
    def to_path(self) -> Path:
        self_asdict: dict[str, Any] = attrs.asdict(self)
        path: Path = Path(self.path_name)
        path /= self.compound.to_path
        for name, attr in attrs.fields_dict(type(self)).items():
            if attr.metadata.get("dir_name", None) is not None:
                val: str = re.sub(r"[^\w\-.]", "_", str(self_asdict[name]))
                path /= f"{attr.metadata['dir_name']}_{val.replace('.', 'p')}"
        return path

    def populate_metadata(self) -> None:
        super().populate_metadata()
        self.metadata["args"]["realizs"] = self.realizs
        self.metadata["args"]["compound"] = RMT_CONVERTER.unstructure(self.compound)

    def realize_monte_carlo_simulation(self) -> None:
        histograms: list[Histogram] = [
            observable.data for observable in self.width_histograms
        ]

        for partial_widths in self.compound.partial_widths_stream(realizs=self.realizs):
            for histogram in histograms:
                width_index: tuple[int, ...] = histogram.metadata["index"]
                width_value: float = compute_width_value(partial_widths, width_index)

                histogram.add_histogram_contribution(width_value)
                histogram.metadata["average_width"] += width_value

        for histogram in histograms:
            histogram.metadata["average_width"] /= self.realizs
            histogram.bins[:] /= histogram.metadata["average_width"]
