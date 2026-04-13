from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any

import numpy as np
from attrs import asdict, frozen, field, fields_dict
from attrs.validators import instance_of, gt
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks
from scipy.special import jn_zeros

from .resonance_histogram import ResonanceHistogramPlot, UnfoldedResonanceHistogramPlot
from .resonance_spacing_histogram import (
    ResonanceSpacingHistogramPlot,
    UnfoldedResonanceSpacingHistogramPlot,
)
from .width_histogram import WidthHistogramPlot, UnfoldedWidthHistogramPlot
from .._histogram import Histogram
from .._simulation import Simulation
from ...compounds import Compound
from ...ensembles import ManyBodyEnsemble
from ...utils import rmtpy_converter


def run_resonance_statistics(compound: Compound, realizs: int) -> None:
    if not isinstance(compound, Compound):
        raise TypeError("compound must be a instance of Compound.")
    elif not isinstance(realizs, int) or realizs <= 0:
        raise ValueError("Number of realizations must be a positive integer.")
    ResonanceStatisticsSimulation(compound=compound, realizs=realizs).run()


@frozen(kw_only=True, repr=False, eq=False, weakref_slot=False, getstate_setstate=False)
class ResonanceStatisticsSimulation(Simulation):
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

    resonance_plot: ResonanceHistogramPlot | None = field(init=False, default=None)
    resonance_support: tuple[float, float] = field(
        default=(-1.2, 1.2)
    )  # units of ground state energy
    resonance_histogram: Histogram = field()

    @resonance_histogram.default
    def _default_resonance_histogram(self) -> Histogram:
        return Histogram(
            file_name="resonance_histogram",
            support=self.resonance_support,
            scale=self.compound.ensemble.ground_state_energy,
        )

    unfolded_resonance_plot: UnfoldedResonanceHistogramPlot | None = field(
        init=False, default=None
    )
    unfolded_resonance_support: tuple[float, float] = field(
        default=(-0.6, 0.6)
    )  # units of dimension
    unfolded_resonance_histogram: Histogram = field()

    @unfolded_resonance_histogram.default
    def _default_unfolded_resonance_histogram(self) -> Histogram:
        return Histogram(
            file_name="unfolded_resonance_histogram",
            support=self.unfolded_resonance_support,
            scale=self.compound.ensemble.dimension,
        )

    width_plot: WidthHistogramPlot | None = field(init=False, default=None)
    width_support: tuple[float, float] = field(default=(-4.0, 4.0))  # log scale base 10
    width_histogram: Histogram = field()

    @width_histogram.default
    def _default_width_histogram(self) -> Histogram:
        histogram: Histogram = Histogram(
            file_name="width_histogram",
            support=self.width_support,
            num_bins=200,
        )
        histogram.bins[:] = np.logspace(
            self.width_support[0],
            self.width_support[1],
            histogram.num_bins + 1,
        )
        return histogram

    unfolded_width_plot: UnfoldedWidthHistogramPlot | None = field(
        init=False, default=None
    )
    unfolded_width_support: tuple[float, float] = field(
        default=(-4.0, 4.0)
    )  # log scale base dimension
    unfolded_width_histogram: Histogram = field()

    @unfolded_width_histogram.default
    def _default_unfolded_width_histogram(self) -> Histogram:
        histogram: Histogram = Histogram(
            file_name="unfolded_width_histogram",
            support=self.unfolded_width_support,
            num_bins=200,
        )
        histogram.bins[:] = np.logspace(
            self.unfolded_width_support[0],
            self.unfolded_width_support[1],
            histogram.num_bins + 1,
        )
        return histogram

    resonance_spacing_plot: ResonanceSpacingHistogramPlot | None = field(
        init=False, default=None
    )
    resonance_spacing_support: tuple[float, float] = field(
        default=(0.0, 4.0)
    )  # units of global mean spacing
    resonance_spacing_histogram: Histogram = field()

    @resonance_spacing_histogram.default
    def _default_resonance_spacing_histogram(self) -> Histogram:
        global_mean_spacing: float = (
            2 * self.compound.ensemble.ground_state_energy
        ) / self.compound.ensemble.dimension

        spacing_histogram = Histogram(
            file_name="resonance_spacing_histogram",
            support=self.resonance_spacing_support,
            scale=global_mean_spacing,
        )
        spacing_histogram.metadata["global_mean_spacing"] = global_mean_spacing
        return spacing_histogram

    unfolded_resonance_spacing_plot: UnfoldedResonanceSpacingHistogramPlot | None = (
        field(init=False, default=None)
    )
    unfolded_resonance_spacing_support: tuple[float, float] = field(default=(0.0, 4.0))
    unfolded_resonance_spacing_histogram: Histogram = field()

    @unfolded_resonance_spacing_histogram.default
    def _default_unfolded_resonance_spacing_histogram(self) -> Histogram:
        return Histogram(
            file_name="unfolded_resonance_spacing_histogram",
            support=self.unfolded_resonance_spacing_support,
        )

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

    def initialize_plots(self) -> None:
        object.__setattr__(
            self,
            "resonance_plot",
            ResonanceHistogramPlot(data=self.resonance_histogram),
        )
        object.__setattr__(
            self,
            "unfolded_resonance_plot",
            UnfoldedResonanceHistogramPlot(data=self.unfolded_resonance_histogram),
        )
        object.__setattr__(
            self,
            "width_plot",
            WidthHistogramPlot(data=self.width_histogram),
        )
        object.__setattr__(
            self,
            "unfolded_width_plot",
            UnfoldedWidthHistogramPlot(data=self.unfolded_width_histogram),
        )
        object.__setattr__(
            self,
            "resonance_spacing_plot",
            ResonanceSpacingHistogramPlot(data=self.resonance_spacing_histogram),
        )
        object.__setattr__(
            self,
            "unfolded_resonance_spacing_plot",
            UnfoldedResonanceSpacingHistogramPlot(
                data=self.unfolded_resonance_spacing_histogram
            ),
        )

    def realize_monte_carlo(self) -> None:
        compound: Compound = self.compound
        ensemble: ManyBodyEnsemble = self.compound.ensemble
        dimension: int = ensemble.dimension
        resonance_histogram: Histogram = self.resonance_histogram
        width_histogram: Histogram = self.width_histogram
        spacing_histogram: Histogram = self.resonance_spacing_histogram
        unfolded_resonance_histogram: Histogram = self.unfolded_resonance_histogram
        unfolded_width_histogram: Histogram = self.unfolded_width_histogram
        unfolded_spacing_histogram: Histogram = (
            self.unfolded_resonance_spacing_histogram
        )

        use_resonance_histogram_cdf: bool = ensemble.num_majoranas >= 26

        for complex_energies in compound.resonances_stream(self.realizs):
            resonances: np.ndarray = complex_energies.real
            widths: np.ndarray = -2 * complex_energies.imag

            resonance_histogram.add_histogram_contribution(resonances)

            width_histogram.add_histogram_contribution(
                widths * compound.resonance_pdf(resonances)
            )

            neighbor_spacings: np.ndarray = np.diff(np.sort(resonances))
            spacing_histogram.add_histogram_contribution(neighbor_spacings)

            if use_resonance_histogram_cdf:
                unfolded_resonances: np.ndarray = resonance_histogram.unfold(
                    dimension, resonances
                )
                unfolded_widths: np.ndarray = resonance_histogram.unfold_locally(
                    dimension, resonances, widths
                )
            else:
                unfolded_resonances: np.ndarray = compound.unfold(resonances)
                unfolded_widths: np.ndarray = (
                    compound.unfold_locally(resonances, widths) / dimension
                )

            unfolded_resonance_histogram.add_histogram_contribution(unfolded_resonances)

            unfolded_width_histogram.add_histogram_contribution(unfolded_widths)

            neighbor_spacings: np.ndarray = np.diff(np.sort(unfolded_resonances))
            unfolded_spacing_histogram.add_histogram_contribution(neighbor_spacings)

    def calculate_statistics(self) -> None:
        self.resonance_histogram.compute_histogram()
        self.width_histogram.compute_histogram()
        self.resonance_spacing_histogram.compute_histogram()

        self.unfolded_resonance_histogram.compute_histogram()
        self.unfolded_width_histogram.compute_histogram()
        self.unfolded_resonance_spacing_histogram.compute_histogram()

    def run(self, out_dir: str | Path = "output") -> None:
        self.realize_monte_carlo()
        self.calculate_statistics()

        out_dir: Path = Path(out_dir)
        base_dir: Path = out_dir / self.to_path
        base_dir.mkdir(parents=True, exist_ok=True)

        self.save_data(out_dir=base_dir)
        self.save_plots(out_dir=base_dir)
