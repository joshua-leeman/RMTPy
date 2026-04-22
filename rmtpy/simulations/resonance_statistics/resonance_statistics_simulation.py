from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any

import numpy as np
from attrs import asdict, frozen, field, fields_dict
from attrs.validators import instance_of, gt
from scipy.special import jn_zeros

from .complex_energy_histogram import (
    ComplexEnergyHistogramPlot,
    UnfoldedComplexEnergyHistogramPlot,
)
from .resonance_histogram import ResonanceHistogramPlot, UnfoldedResonanceHistogramPlot
from .resonance_spacing_histogram import (
    ResonanceSpacingHistogramPlot,
    UnfoldedResonanceSpacingHistogramPlot,
)
from .width_histogram import WidthHistogramPlot, UnfoldedWidthHistogramPlot
from .resonance_form_factors import (
    FormFactorsData,
    ResonanceFormFactorsPlot,
    UnfoldedResonanceFormFactorsPlot,
)
from .._histogram import Histogram, Histogram2D
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
        default=(-2.0, 2.0)
    )  # units of ground state energy
    resonance_histogram: Histogram = field()

    @resonance_histogram.default
    def _default_resonance_histogram(self) -> Histogram:
        return Histogram(
            file_name="resonance_histogram",
            support=self.resonance_support,
            scale=self.compound.ensemble.ground_state_energy,
            num_bins=200,
        )

    unfolded_resonance_plot: UnfoldedResonanceHistogramPlot | None = field(
        init=False, default=None
    )
    unfolded_resonance_support: tuple[float, float] = field(
        default=(-1.0, 1.0)
    )  # units of dimension
    unfolded_resonance_histogram: Histogram = field()

    @unfolded_resonance_histogram.default
    def _default_unfolded_resonance_histogram(self) -> Histogram:
        return Histogram(
            file_name="unfolded_resonance_histogram",
            support=self.unfolded_resonance_support,
            scale=self.compound.ensemble.dimension,
            num_bins=200,
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
        histogram.bins[:] = histogram.scale * np.logspace(
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
        default=(0.0, 24.0)
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
            num_bins=600,
        )
        spacing_histogram.metadata["global_mean_spacing"] = global_mean_spacing
        return spacing_histogram

    unfolded_resonance_spacing_plot: UnfoldedResonanceSpacingHistogramPlot | None = (
        field(init=False, default=None)
    )
    unfolded_resonance_spacing_support: tuple[float, float] = field(default=(0.0, 24.0))
    unfolded_resonance_spacing_histogram: Histogram = field()

    @unfolded_resonance_spacing_histogram.default
    def _default_unfolded_resonance_spacing_histogram(self) -> Histogram:
        return Histogram(
            file_name="unfolded_resonance_spacing_histogram",
            support=self.unfolded_resonance_spacing_support,
            num_bins=600,
        )

    complex_energy_plot: ComplexEnergyHistogramPlot | None = field(
        init=False, default=None
    )
    complex_energy_support: tuple[tuple[float, float], tuple[float, float]] = field(
        default=((-2.0, 2.0), (-8.0, 8.0))
    )
    complex_energy_histogram: Histogram2D = field()

    @complex_energy_histogram.default
    def _default_complex_energy_histogram(self) -> Histogram2D:
        histogram: Histogram2D = Histogram2D(
            file_name="complex_energy_histogram",
            x_support=self.complex_energy_support[0],
            y_support=self.complex_energy_support[1],
            x_num_bins=400,
            y_num_bins=400,
        )
        histogram.y_bins[:] = np.logspace(
            self.complex_energy_support[1][0],
            self.complex_energy_support[1][1],
            histogram.y_num_bins + 1,
        )
        return histogram

    unfolded_complex_energy_plot: UnfoldedComplexEnergyHistogramPlot | None = field(
        init=False, default=None
    )
    unfolded_complex_energy_support: tuple[tuple[float, float], tuple[float, float]] = (
        field(default=((-2.0, 2.0), (-8.0, 8.0)))
    )
    unfolded_complex_energy_histogram: Histogram2D = field()

    @unfolded_complex_energy_histogram.default
    def _default_unfolded_complex_energy_histogram(self) -> Histogram2D:
        histogram: Histogram2D = Histogram2D(
            file_name="unfolded_complex_energy_histogram",
            x_support=self.unfolded_complex_energy_support[0],
            y_support=self.unfolded_complex_energy_support[1],
            x_num_bins=400,
            y_num_bins=400,
        )
        histogram.y_bins[:] = np.logspace(
            self.unfolded_complex_energy_support[1][0],
            self.unfolded_complex_energy_support[1][1],
            histogram.y_num_bins + 1,
        )
        return histogram

    form_factors_plot: ResonanceFormFactorsPlot | None = field(init=False, default=None)
    form_factors_support: tuple[float, float] = field(
        default=(-0.5, 1.5)
    )  # log time base dimension
    form_factors_data: FormFactorsData = field()

    @form_factors_data.default
    def _default_form_factors_data(self) -> FormFactorsData:
        j_1_1: float = float(jn_zeros(1, 1)[0])
        return FormFactorsData(
            file_name="resonance_form_factors",
            dimension=self.compound.ensemble.dimension,
            log_D_time_support=self.form_factors_support,
            scale=j_1_1 / self.compound.ensemble.ground_state_energy,
        )

    unfolded_form_factors_plot: UnfoldedResonanceFormFactorsPlot | None = field(
        init=False, default=None
    )
    unfolded_form_factors_support: tuple[float, float] = field(
        default=(-1.5, 0.5)
    )  # log time base dimension
    unfolded_form_factors_data: FormFactorsData = field()

    @unfolded_form_factors_data.default
    def _default_unfolded_form_factors_data(self) -> FormFactorsData:
        return FormFactorsData(
            file_name="unfolded_resonance_form_factors",
            dimension=self.compound.ensemble.dimension,
            log_D_time_support=self.unfolded_form_factors_support,
            scale=2 * np.pi,
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
        object.__setattr__(
            self,
            "complex_energy_plot",
            ComplexEnergyHistogramPlot(data=self.complex_energy_histogram),
        )
        object.__setattr__(
            self,
            "unfolded_complex_energy_plot",
            UnfoldedComplexEnergyHistogramPlot(
                data=self.unfolded_complex_energy_histogram
            ),
        )
        object.__setattr__(
            self,
            "form_factors_plot",
            ResonanceFormFactorsPlot(data=self.form_factors_data),
        )
        object.__setattr__(
            self,
            "unfolded_form_factors_plot",
            UnfoldedResonanceFormFactorsPlot(data=self.unfolded_form_factors_data),
        )

    def realize_monte_carlo(self) -> None:
        compound: Compound = self.compound
        ensemble: ManyBodyEnsemble = self.compound.ensemble
        dimension: int = ensemble.dimension
        energy_0: float = ensemble.ground_state_energy

        resonance_histogram: Histogram = self.resonance_histogram
        width_histogram: Histogram = self.width_histogram
        spacing_histogram: Histogram = self.resonance_spacing_histogram
        complex_energy_histogram: Histogram2D = self.complex_energy_histogram
        form_factors_data: FormFactorsData = self.form_factors_data
        unfolded_resonance_histogram: Histogram = self.unfolded_resonance_histogram
        unfolded_width_histogram: Histogram = self.unfolded_width_histogram
        unfolded_spacing_histogram: Histogram = (
            self.unfolded_resonance_spacing_histogram
        )
        unfolded_complex_energy_histogram: Histogram2D = (
            self.unfolded_complex_energy_histogram
        )
        unfolded_form_factors_data: FormFactorsData = self.unfolded_form_factors_data

        use_resonance_histogram_cdf: bool = ensemble.num_majoranas >= 26

        for complex_energies in compound.resonances_stream(self.realizs):
            resonances: np.ndarray = complex_energies.real
            widths: np.ndarray = -2 * complex_energies.imag

            resonance_histogram.add_histogram_contribution(resonances)

            width_histogram.add_histogram_contribution(widths / energy_0)

            neighbor_spacings: np.ndarray = np.diff(np.sort(resonances))
            spacing_histogram.add_histogram_contribution(neighbor_spacings)

            complex_energy_histogram.add_histogram_contribution(
                resonances / energy_0, widths / energy_0
            )

            form_factors_data.compute_moment_contributions(resonances)

            if use_resonance_histogram_cdf:
                unfolded_resonances: np.ndarray = resonance_histogram.unfold(
                    dimension, resonances
                )
                unfolded_widths: np.ndarray = resonance_histogram.unfold_widths(
                    dimension, resonances, widths
                )
            else:
                unfolded_resonances: np.ndarray = compound.unfold(resonances)
                unfolded_widths: np.ndarray = compound.unfold_widths(resonances, widths)

            unfolded_resonance_histogram.add_histogram_contribution(unfolded_resonances)

            unfolded_width_histogram.add_histogram_contribution(unfolded_widths)

            neighbor_spacings: np.ndarray = np.diff(np.sort(unfolded_resonances))
            unfolded_spacing_histogram.add_histogram_contribution(neighbor_spacings)

            unfolded_complex_energy_histogram.add_histogram_contribution(
                resonances / energy_0, unfolded_widths
            )

            unfolded_form_factors_data.compute_moment_contributions(unfolded_resonances)

    def calculate_statistics(self) -> None:
        self.resonance_histogram.compute_histogram_density()
        self.width_histogram.compute_histogram_density()
        self.resonance_spacing_histogram.compute_histogram_density()
        self.complex_energy_histogram.compute_histogram_probabilities()
        self.form_factors_data.compute_form_factors()

        self.unfolded_resonance_histogram.compute_histogram_density()
        self.unfolded_width_histogram.compute_histogram_density()
        self.unfolded_resonance_spacing_histogram.compute_histogram_density()
        self.unfolded_complex_energy_histogram.compute_histogram_probabilities()
        self.unfolded_form_factors_data.compute_form_factors()

    def run(self, out_dir: str | Path = "output") -> None:
        self.realize_monte_carlo()
        self.calculate_statistics()

        out_dir: Path = Path(out_dir)
        base_dir: Path = out_dir / self.to_path
        base_dir.mkdir(parents=True, exist_ok=True)

        self.save_data(out_dir=base_dir)
        self.save_plots(out_dir=base_dir)
