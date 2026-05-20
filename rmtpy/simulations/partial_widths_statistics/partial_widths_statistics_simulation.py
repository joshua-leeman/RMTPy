from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any

import numpy as np
from attrs import asdict, field, fields_dict, frozen
from attrs.validators import gt

from ...compounds import Compound
from ...conversion import rmtpy_converter
from ..histogram import Histogram
from .partial_width_histogram import PartialWidthHistogramPlot, TotalWidthHistogramPlot
from ..base import Simulation


def run_partial_widths_statistics(compound: Compound, realizs: int) -> None:
    if not isinstance(compound, Compound):
        raise TypeError("compound must be a instance of Compound.")
    elif not isinstance(realizs, int) or realizs <= 0:
        raise ValueError("Number of realizations must be a positive integer.")
    PartialWidthsStatisticsSimulation(compound=compound, realizs=realizs).run()


@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class PartialWidthsStatisticsSimulation(Simulation):
    width_indices: tuple[tuple[int, ...]] = field(
        converter=tuple, default=((0, 0), (1, 0), (1, 1), (0,), (1,))
    )

    realizs: int = field(
        converter=int,
        validator=gt(0),
        metadata={"dir_name": "realizs", "latex_name": "R"},
    )

    compound: Compound = field(converter=Compound.create)

    @compound.validator
    def _compound_validator(self, _, value: Compound) -> None:
        if inspect.isabstract(value):
            raise ValueError(f"Compound must be concrete.")

    partial_width_support: tuple[float, float] = field(
        default=(-5.0, 2.0)
    )  # log scale base 10

    total_width_support: tuple[float, float] = field(
        default=(-2.0, 3.0)
    )  # log scale base 10

    width_histograms: list[Histogram] = field(init=False, repr=False)

    @width_histograms.default
    def _default_width_histograms(self) -> list[Histogram]:
        histograms: list[Histogram] = []
        for width_index in self.width_indices:
            if len(width_index) == 2:
                histogram: Histogram = Histogram(
                    file_name=f"partial_width_{width_index[0]}{width_index[1]}_histogram",
                    log_base=10.0,
                    support=self.partial_width_support,
                    num_bins=60,
                )
            elif len(width_index) == 1:
                histogram: Histogram = Histogram(
                    file_name=f"total_width_{width_index[0]}_histogram",
                    log_base=10.0,
                    support=self.total_width_support,
                    num_bins=100,
                )
            else:
                raise ValueError("Invalid width index in width_indices.")

            histogram.metadata["index"] = width_index
            histogram.metadata["average_width"] = 0.0
            histograms.append(histogram)

        return histograms

    width_histogram_plots: list[PartialWidthHistogramPlot | TotalWidthHistogramPlot] = (
        field(init=False, repr=False, factory=list)
    )

    @property
    def to_path(self) -> Path:
        self_asdict: dict[str, Any] = asdict(self)
        path: Path = Path(self.path_name)
        path /= self.compound.to_path
        for name, attr in fields_dict(type(self)).items():
            if attr.metadata.get("dir_name", None) is not None:
                val: str = re.sub(r"[^\w\-.]", "_", str(self_asdict[name]))
                path /= f"{attr.metadata['dir_name']}_{val.replace('.', 'p')}"
        return path

    def populate_metadata(self) -> None:
        super().populate_metadata()
        self.metadata["args"]["realizs"] = self.realizs
        self.metadata["args"]["compound"] = rmtpy_converter.unstructure(self.compound)

    def initialize_plots(self) -> None:
        for histogram in self.width_histograms:
            if len(histogram.metadata["index"]) == 2:
                self.width_histogram_plots.append(
                    PartialWidthHistogramPlot(data=histogram)
                )
            elif len(histogram.metadata["index"]) == 1:
                self.width_histogram_plots.append(
                    TotalWidthHistogramPlot(data=histogram)
                )
            else:
                raise ValueError("Invalid width index in histogram metadata.")

    def realize_monte_carlo(self) -> None:
        for partial_widths in self.compound.partial_widths_stream(realizs=self.realizs):
            for histogram in self.width_histograms:
                width_index: tuple[int, ...] = histogram.metadata["index"]
                if len(width_index) == 2:
                    width_value: float = partial_widths[width_index[0]][width_index[1]]
                else:
                    width_value: float = np.sum(partial_widths[width_index[0]])

                histogram.add_histogram_contribution(width_value)
                histogram.metadata["average_width"] += width_value

        for histogram in self.width_histograms:
            histogram.metadata["average_width"] /= self.realizs
            histogram.bins[:] /= histogram.metadata["average_width"]

    def calculate_statistics(self) -> None:
        for histogram in self.width_histograms:
            histogram.compute_histogram_density()

    def run(self, out_dir: str | Path = "output") -> None:
        self.realize_monte_carlo()
        self.calculate_statistics()

        out_dir: Path = Path(out_dir)
        base_dir: Path = out_dir / self.to_path
        base_dir.mkdir(parents=True, exist_ok=True)

        self.save_data(out_dir=base_dir)
        self.save_plots(out_dir=base_dir)
