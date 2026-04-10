from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
from attrs import frozen, field
from attrs.validators import instance_of
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks
from scipy.special import jn_zeros

from .spectral_histogram import SpectralHistogramPlot
from .spectral_histogram import UnfoldedSpectralHistogramPlot
from .spacings_histogram import SpacingsHistogramPlot
from .spacings_histogram import UnfoldedSpacingsHistogramPlot
from .spectral_form_factors import FormFactorsPlot
from .spectral_form_factors import UnfoldedFormFactorsPlot
from .spectral_form_factors import FormFactorsData
from .._histogram import Histogram
from .._simulation import Simulation
from ...ensembles import ManyBodyEnsemble


def thouless_time(times: np.ndarray, form_factor: np.ndarray) -> float:
    max_idx, _ = find_peaks(form_factor)

    pchip = PchipInterpolator(times[max_idx], form_factor[max_idx])

    start = max_idx[0]
    stop = max_idx[-1] + 1
    envelope = pchip(times[start:stop])
    relative_min_idx = np.argmin(envelope)
    thouless_idx = np.searchsorted(times, times[start:stop][relative_min_idx])

    return float(times[thouless_idx])


def run_spectral_statistics(ensemble: ManyBodyEnsemble, realizs: int) -> None:
    if not isinstance(ensemble, ManyBodyEnsemble):
        raise TypeError("Ensemble must be an instance of ManyBodyEnsemble.")
    elif not isinstance(realizs, int) or realizs <= 0:
        raise ValueError("Number of realizations must be a positive integer.")
    SpectralStatisticsSimulation(ensemble=ensemble, realizs=realizs).run()


@frozen(kw_only=True, repr=False, eq=False, weakref_slot=False, getstate_setstate=False)
class SpectralStatisticsSimulation(Simulation):
    ensemble: ManyBodyEnsemble = field(
        converter=ManyBodyEnsemble.create, validator=instance_of(ManyBodyEnsemble)
    )

    @ensemble.validator
    def _ensemble_validator(self, _, value: ManyBodyEnsemble) -> None:
        if inspect.isabstract(value):
            raise ValueError(f"ManyBodyEnsemble must be concrete.")

    spectral_plot: SpectralHistogramPlot | None = field(init=False, default=None)
    spectral_support: tuple[float, float] = field(
        default=(-1.2, 1.2)
    )  # units of ground state energy
    spectral_histogram: Histogram = field()

    @spectral_histogram.default
    def _default_spectral_histogram(self) -> Histogram:
        return Histogram(
            file_name="spectral_histogram",
            support=self.spectral_support,
            scale=self.ensemble.ground_state_energy,
        )

    unfolded_spectral_plot: UnfoldedSpectralHistogramPlot | None = field(
        init=False, default=None
    )
    unfolded_spectral_support: tuple[float, float] = field(
        default=(-0.6, 0.6)
    )  # units of dimension
    unfolded_spectral_histogram: Histogram = field()

    @unfolded_spectral_histogram.default
    def _default_unfolded_spectral_histogram(self) -> Histogram:
        return Histogram(
            file_name="unfolded_spectral_histogram",
            support=self.unfolded_spectral_support,
            scale=self.ensemble.dimension,
        )

    spacings_plot: SpacingsHistogramPlot | None = field(init=False, default=None)
    spacings_support: tuple[float, float] = field(
        default=(0.0, 4.0)
    )  # units of global mean spacing
    spacings_histogram: Histogram = field()

    @spacings_histogram.default
    def _default_spacings_histogram(self) -> Histogram:
        global_mean_spacing: float = (
            2 * self.ensemble.ground_state_energy / self.ensemble.dimension
        )
        spacings_histogram = Histogram(
            file_name="spacings_histogram",
            support=self.spacings_support,
            scale=global_mean_spacing,
        )
        spacings_histogram.metadata["global_mean_spacing"] = global_mean_spacing
        return spacings_histogram

    unfolded_spacings_plot: UnfoldedSpacingsHistogramPlot | None = field(
        init=False, default=None
    )
    unfolded_spacings_support: tuple[float, float] = field(default=(0.0, 4.0))
    unfolded_spacings_histogram: Histogram = field()

    @unfolded_spacings_histogram.default
    def _default_unfolded_spacings_histogram(self) -> Histogram:
        return Histogram(
            file_name="unfolded_spacings_histogram",
            support=self.unfolded_spacings_support,
        )

    form_factors_plot: FormFactorsPlot | None = field(init=False, default=None)
    form_factors_support: tuple[float, float] = field(
        default=(-0.5, 1.5)
    )  # log time base dimension
    form_factors_data: FormFactorsData = field()

    @form_factors_data.default
    def _default_form_factors_data(self) -> FormFactorsData:
        j_1_1: float = float(jn_zeros(1, 1)[0])
        return FormFactorsData(
            file_name="spectral_form_factors",
            dimension=self.ensemble.dimension,
            log_D_time_support=self.form_factors_support,
            scale=j_1_1 / self.ensemble.ground_state_energy,
        )

    unfolded_form_factors_plot: UnfoldedFormFactorsPlot | None = field(
        init=False, default=None
    )
    unfolded_form_factors_support: tuple[float, float] = field(
        default=(-1.5, 0.5)
    )  # log time base dimension
    unfolded_form_factors_data: FormFactorsData = field()

    @unfolded_form_factors_data.default
    def _default_unfolded_form_factors_data(self) -> FormFactorsData:
        return FormFactorsData(
            file_name="unfolded_spectral_form_factors",
            dimension=self.ensemble.dimension,
            log_D_time_support=self.unfolded_form_factors_support,
            scale=2 * np.pi,
        )

    def initialize_plots(self) -> None:
        object.__setattr__(
            self,
            "spectral_plot",
            SpectralHistogramPlot(data=self.spectral_histogram),
        )
        object.__setattr__(
            self,
            "unfolded_spectral_plot",
            UnfoldedSpectralHistogramPlot(data=self.unfolded_spectral_histogram),
        )
        object.__setattr__(
            self,
            "spacings_plot",
            SpacingsHistogramPlot(data=self.spacings_histogram),
        )
        object.__setattr__(
            self,
            "unfolded_spacings_plot",
            UnfoldedSpacingsHistogramPlot(data=self.unfolded_spacings_histogram),
        )
        object.__setattr__(
            self,
            "form_factors_plot",
            FormFactorsPlot(data=self.form_factors_data),
        )
        object.__setattr__(
            self,
            "unfolded_form_factors_plot",
            UnfoldedFormFactorsPlot(data=self.unfolded_form_factors_data),
        )

    def realize_monte_carlo(self) -> None:
        degeneracy: int = self.ensemble.eigval_degeneracy
        spectral_histogram: Histogram = self.spectral_histogram
        spacings_histogram: Histogram = self.spacings_histogram
        form_factors_data: FormFactorsData = self.form_factors_data
        unfolded_spectral_histogram: Histogram = self.unfolded_spectral_histogram
        unfolded_spacings_histogram: Histogram = self.unfolded_spacings_histogram
        unfolded_form_factors_data: FormFactorsData = self.unfolded_form_factors_data

        for eigvals in self.ensemble.eigvals_stream(self.realizs):
            spectral_histogram.add_histogram_contribution(eigvals)
            neighbor_spacings = np.diff(np.sort(eigvals))
            if degeneracy > 1:
                neighbor_spacings = np.repeat(
                    neighbor_spacings[1::degeneracy], degeneracy
                )
            spacings_histogram.add_histogram_contribution(neighbor_spacings)
            form_factors_data.compute_moment_contributions(eigvals)

            unfolded_eigvals = self.ensemble.unfold(eigvals)

            unfolded_spectral_histogram.add_histogram_contribution(unfolded_eigvals)
            neighbor_spacings = np.diff(np.sort(unfolded_eigvals))
            if degeneracy > 1:
                neighbor_spacings = np.repeat(
                    neighbor_spacings[1::degeneracy], degeneracy
                )
            unfolded_spacings_histogram.add_histogram_contribution(neighbor_spacings)
            unfolded_form_factors_data.compute_moment_contributions(unfolded_eigvals)

    def calculate_statistics(self) -> None:
        self.spectral_histogram.compute_histogram()
        self.spacings_histogram.compute_histogram()
        self.form_factors_data.compute_form_factors()

        self.unfolded_spectral_histogram.compute_histogram()
        self.unfolded_spacings_histogram.compute_histogram()
        self.unfolded_form_factors_data.compute_form_factors()

    def run(self, out_dir: str | Path = "output") -> None:
        self.realize_monte_carlo()
        self.calculate_statistics()

        out_dir: Path = Path(out_dir)
        base_dir: Path = out_dir / self.to_dir
        base_dir.mkdir(parents=True, exist_ok=True)

        self.save_data(out_dir=base_dir)
        self.save_plots(out_dir=base_dir)
