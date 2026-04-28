from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any

import numpy as np
from attrs import asdict, frozen, field, fields_dict
from attrs.validators import instance_of, gt

from .partial_width_histogram import PartialWidthHistogramPlot
from .partial_width_histogram import TotalWidthHistogramPlot
from .._histogram import Histogram
from .._simulation import Simulation
from ...compounds import Compound
from ...utils import rmtpy_converter


def run_partial_width_statistics(compound: Compound, realizs: int) -> None:
    if not isinstance(compound, Compound):
        raise TypeError("compound must be a instance of Compound.")
    elif not isinstance(realizs, int) or realizs <= 0:
        raise ValueError("Number of realizations must be a positive integer.")
    PartialWidthStatisticsSimulation(compound=compound, realizs=realizs).run()


@frozen(kw_only=True, repr=False, eq=False, weakref_slot=False, getstate_setstate=False)
class PartialWidthStatisticsSimulation(Simulation):
    realizs: int = field(
        converter=int,
        validator=gt(0),
        metadata={"dir_name": "realizs", "latex_name": "R"},
    )

    compound: Compound = field(
        converter=Compound.create, validator=instance_of(Compound)
    )

    @compound.validator
    def _compound_validator(self, _, value: Compound) -> None:
        if inspect.isabstract(value):
            raise ValueError(f"Compound must be concrete.")

    partial_widths_support: tuple[float, float] = field(
        default=(-5.0, 2.0)
    )  # log scale base 10

    partial_width_00_histogram_plot: PartialWidthHistogramPlot | None = field(
        init=False, default=None
    )
    partial_width_00_histogram: Histogram = field()

    @partial_width_00_histogram.default
    def _default_partial_width_00_histogram(self) -> Histogram:
        histogram: Histogram = Histogram(
            file_name="partial_width_00_histogram",
            log_base=10.0,
            support=self.partial_widths_support,
            num_bins=60,
        )
        histogram.metadata["index"] = (0, 0)
        histogram.metadata["ave_width"] = 0.0
        return histogram

    partial_width_10_histogram_plot: PartialWidthHistogramPlot | None = field(
        init=False, default=None
    )
    partial_width_10_histogram: Histogram = field()

    @partial_width_10_histogram.default
    def _default_partial_width_10_histogram(self) -> Histogram:
        histogram: Histogram = Histogram(
            file_name="partial_width_10_histogram",
            log_base=10.0,
            support=self.partial_widths_support,
            num_bins=60,
        )
        histogram.metadata["index"] = (1, 0)
        histogram.metadata["ave_width"] = 0.0
        return histogram

    partial_width_11_histogram_plot: PartialWidthHistogramPlot | None = field(
        init=False, default=None
    )
    partial_width_11_histogram: Histogram = field()

    @partial_width_11_histogram.default
    def _default_partial_width_11_histogram(self) -> Histogram:
        histogram: Histogram = Histogram(
            file_name="partial_width_11_histogram",
            log_base=10.0,
            support=self.partial_widths_support,
            num_bins=60,
        )
        histogram.metadata["index"] = (1, 1)
        histogram.metadata["ave_width"] = 0.0
        return histogram

    total_widths_support: tuple[float, float] = field(
        default=(-2.0, 3.0)
    )  # log scale base 10

    total_width_0_histogram_plot: PartialWidthHistogramPlot | None = field(
        init=False, default=None
    )
    total_width_0_histogram: Histogram = field()

    @total_width_0_histogram.default
    def _default_total_width_0_histogram(self) -> Histogram:
        histogram: Histogram = Histogram(
            file_name="total_width_0_histogram",
            log_base=10.0,
            support=self.total_widths_support,
            num_bins=100,
        )
        histogram.metadata["index"] = (0,)
        histogram.metadata["ave_width"] = 0.0
        return histogram

    total_width_1_histogram_plot: PartialWidthHistogramPlot | None = field(
        init=False, default=None
    )
    total_width_1_histogram: Histogram = field()

    @total_width_1_histogram.default
    def _default_total_width_1_histogram(self) -> Histogram:
        histogram: Histogram = Histogram(
            file_name="total_width_1_histogram",
            log_base=10.0,
            support=self.total_widths_support,
            num_bins=100,
        )
        histogram.metadata["index"] = (1,)
        histogram.metadata["ave_width"] = 0.0
        return histogram

    width_histograms: list[Histogram] = field(init=False, repr=False, factory=list)

    @property
    def to_path(self) -> Path:
        self_asdict: dict[str, Any] = asdict(self)
        path: Path = Path(self._path_name)
        path /= self.compound.to_path
        for name, attr in fields_dict(type(self)).items():
            if attr.metadata.get("dir_name", None) is not None:
                val: str = re.sub(r"[^\w\-.]", "_", str(self_asdict[name]))
                path /= f"{attr.metadata['dir_name']}_{val.replace('.', 'p')}"
        return path

    def _populate_metadata(self) -> None:
        super()._populate_metadata()
        self.metadata["args"]["realizs"] = self.realizs
        self.metadata["args"]["compound"] = rmtpy_converter.unstructure(self.compound)

        for attribute_name in asdict(self).keys():
            attribute_value = getattr(self, attribute_name)
            if type(attribute_value) == Histogram:
                self.width_histograms.append(attribute_value)

    def initialize_plots(self) -> None:
        for histogram in self.width_histograms:
            if len(histogram.metadata["index"]) == 2:
                object.__setattr__(
                    self,
                    histogram.file_name.replace("data", "plot"),
                    PartialWidthHistogramPlot(data=histogram),
                )
            elif len(histogram.metadata["index"]) == 1:
                object.__setattr__(
                    self,
                    histogram.file_name.replace("data", "plot"),
                    TotalWidthHistogramPlot(data=histogram),
                )
            else:
                raise ValueError("Invalid width index in histogram metadata.")

    def realize_monte_carlo(self) -> None:
        width_histograms: list[Histogram] = [
            getattr(self, attribute_name)
            for attribute_name in asdict(self).keys()
            if type(getattr(self, attribute_name)) == Histogram
        ]

        for partial_widths in self.compound.partial_widths_stream(realizs=self.realizs):
            for histogram in width_histograms:
                width_index: tuple[int, ...] = histogram.metadata["index"]
                if len(width_index) == 2:
                    width_value: float = partial_widths[width_index[0]][width_index[1]]
                else:
                    width_value: float = np.sum(partial_widths[width_index[0]])

                histogram.add_histogram_contribution(width_value)
                histogram.metadata["ave_width"] += width_value

        for histogram in width_histograms:
            histogram.metadata["ave_width"] /= self.realizs
            histogram.bins[:] /= histogram.metadata["ave_width"]

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
