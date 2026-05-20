from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any

import attrs
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks
from scipy.special import jn_zeros

import rmtpy.ensembles
from ..histogram import Histogram
from ..observable import Observable
from ..base import Simulation
from .spacings_histogram import (
    SpacingsHistogramPlot,
    UnfoldedSpacingsHistogramPlot,
)
from .spectral_form_factors import (
    FormFactorsData,
    FormFactorsPlot,
    UnfoldedFormFactorsPlot,
)
from .spectral_histogram import (
    SpectralHistogramPlot,
    UnfoldedSpectralHistogramPlot,
)


def thouless_time(times: np.ndarray, form_factor: np.ndarray) -> float:
    max_idx, _ = find_peaks(form_factor)

    pchip = PchipInterpolator(times[max_idx], form_factor[max_idx])

    start = max_idx[0]
    stop = max_idx[-1] + 1
    envelope = pchip(times[start:stop])
    relative_min_idx = np.argmin(envelope)
    thouless_idx = np.searchsorted(times, times[start:stop][relative_min_idx])

    return float(times[thouless_idx])


def run_spectral_statistics(
    ensemble: rmtpy.ensembles.ManyBodyEnsemble, realizs: int
) -> None:
    if not isinstance(ensemble, rmtpy.ensembles.ManyBodyEnsemble):
        raise TypeError("Ensemble must be an instance of ManyBodyEnsemble.")

    SpectralStatisticsSimulation(ensemble=ensemble, realizs=realizs).run()


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class SpectralStatisticsSimulation(Simulation):
    ensemble: rmtpy.ensembles.ManyBodyEnsemble = attrs.field(
        converter=rmtpy.ensembles.ManyBodyEnsemble.create
    )

    @ensemble.validator
    def _ensemble_validator(self, _, value: rmtpy.ensembles.ManyBodyEnsemble) -> None:
        if inspect.isabstract(value):
            raise ValueError("ManyBodyEnsemble must be concrete.")

    realizs: int = attrs.field(
        converter=int,
        validator=attrs.validators.gt(0),
        metadata={"dir_name": "realizs", "latex_name": "R"},
    )

    spectral: Observable = attrs.field()

    @spectral.default
    def _default_spectral(self) -> Observable:
        return Observable(
            data=Histogram(
                file_name="spectral_histogram",
                scale=self.ensemble.spectral_radius,
            ),
            plot_cls=SpectralHistogramPlot,
            finalize=lambda data: data.compute_histogram_density(),
        )

    spectral_avg_unfolded: Observable = attrs.field()

    @spectral_avg_unfolded.default
    def _default_spectral_avg_unfolded(self) -> Observable:
        return Observable(
            data=Histogram(
                file_name="spectral_histogram_avg_unfolded",
                scale=self.ensemble.spectral_radius,
            ),
            plot_cls=UnfoldedSpectralHistogramPlot,
            finalize=lambda data: data.compute_histogram_density(),
        )

    spectral_var_unfolded: Observable = attrs.field()

    @spectral_var_unfolded.default
    def _default_spectral_var_unfolded(self) -> Observable:
        return Observable(
            data=Histogram(
                file_name="spectral_histogram_var_unfolded",
                scale=self.ensemble.spectral_radius,
            ),
            plot_cls=UnfoldedSpectralHistogramPlot,
            finalize=lambda data: data.compute_histogram_density(),
        )

    spacings: Observable = attrs.field()

    @spacings.default
    def _default_spacings(self) -> Observable:
        global_mean_spacing: float = (
            2 * self.ensemble.spectral_radius / self.ensemble.dimension
        )

        spacings_histogram = Histogram(
            file_name="spacings_histogram",
            support=(0.0, 4.0),
            scale=global_mean_spacing,
        )
        spacings_histogram.metadata["global_mean_spacing"] = global_mean_spacing

        return Observable(
            data=spacings_histogram,
            plot_cls=SpacingsHistogramPlot,
            finalize=lambda data: data.compute_histogram_density(),
        )

    spacings_avg_unfolded: Observable = attrs.field()

    @spacings_avg_unfolded.default
    def _default_spacings_avg_unfolded(self) -> Observable:
        return Observable(
            data=Histogram(
                file_name="spacings_histogram_avg_unfolded",
                support=(0.0, 4.0),
            ),
            plot_cls=UnfoldedSpacingsHistogramPlot,
            finalize=lambda data: data.compute_histogram_density(),
        )

    spacings_var_unfolded: Observable = attrs.field()

    @spacings_var_unfolded.default
    def _default_spacings_var_unfolded(self) -> Observable:
        return Observable(
            data=Histogram(
                file_name="spacings_histogram_var_unfolded",
                support=(0.0, 4.0),
            ),
            plot_cls=UnfoldedSpacingsHistogramPlot,
            finalize=lambda data: data.compute_histogram_density(),
        )

    form_factors: Observable = attrs.field()

    @form_factors.default
    def _default_form_factors(self) -> Observable:
        j_1_1: float = float(jn_zeros(1, 1)[0])
        return Observable(
            data=FormFactorsData(
                file_name="spectral_form_factors",
                dimension=self.ensemble.dimension,
                log_D_time_support=(-0.5, 1.5),
                scale=j_1_1 / self.ensemble.spectral_radius,
            ),
            plot_cls=FormFactorsPlot,
            finalize=lambda data: data.compute_form_factors(),
        )

    form_factors_avg_unfolded: Observable = attrs.field()

    @form_factors_avg_unfolded.default
    def _default_form_factors_avg_unfolded(self) -> Observable:
        return Observable(
            data=FormFactorsData(
                file_name="spectral_form_factors_avg_unfolded",
                dimension=self.ensemble.dimension,
                log_D_time_support=(-1.5, 0.5),
                scale=2 * np.pi,
            ),
            plot_cls=UnfoldedFormFactorsPlot,
            finalize=lambda data: data.compute_form_factors(),
        )

    form_factors_var_unfolded: Observable = attrs.field()

    @form_factors_var_unfolded.default
    def _default_form_factors_var_unfolded(self) -> Observable:
        return Observable(
            data=FormFactorsData(
                file_name="spectral_form_factors_var_unfolded",
                dimension=self.ensemble.dimension,
                log_D_time_support=(-1.5, 0.5),
                scale=2 * np.pi,
            ),
            plot_cls=UnfoldedFormFactorsPlot,
            finalize=lambda data: data.compute_form_factors(),
        )

    @property
    def to_path(self) -> Path:
        self_asdict: dict[str, Any] = attrs.asdict(self)
        path: Path = Path(self.path_name)
        path /= self.ensemble.to_path
        for name, attr in attrs.fields_dict(type(self)).items():
            if attr.metadata.get("dir_name", None) is not None:
                val: str = re.sub(r"[^\w\-.]", "_", str(self_asdict[name]))
                path /= f"{attr.metadata['dir_name']}_{val.replace('.', 'p')}"
        return path

    def populate_metadata(self) -> None:
        super().populate_metadata()
        self.metadata["args"]["ensemble"] = rmtpy_converter.unstructure(self.ensemble)
        self.metadata["args"]["realizs"] = self.realizs

    def compute_nearest_neighbor_spacings(self, eigvals: np.ndarray) -> np.ndarray:
        degeneracy: int = self.ensemble.eigval_degeneracy
        nn_spacings: np.ndarray = np.diff(np.sort(eigvals))
        if degeneracy > 1:
            nn_spacings = np.repeat(nn_spacings[1::degeneracy], degeneracy)
        return nn_spacings

    def realize_monte_carlo_simulation(self) -> None:
        degeneracy: int = self.ensemble.eigval_degeneracy

        spectral_histogram: Histogram = self.spectral.data
        spectral_histogram_avg_unfolded: Histogram = self.spectral_avg_unfolded.data
        spectral_histogram_var_unfolded: Histogram = self.spectral_var_unfolded.data

        spacings_histogram: Histogram = self.spacings.data
        spacings_histogram_avg_unfolded: Histogram = self.spacings_avg_unfolded.data
        spacings_histogram_var_unfolded: Histogram = self.spacings_var_unfolded.data

        form_factors: FormFactorsData = self.form_factors.data
        form_factors_avg_unfolded: FormFactorsData = self.form_factors_avg_unfolded.data
        form_factors_var_unfolded: FormFactorsData = self.form_factors_var_unfolded.data

        for eigvals in self.ensemble.eigvals_stream(self.realizs):
            spectral_histogram.add_histogram_contribution(eigvals)
            nn_spacings: np.ndarray = self.compute_nearest_neighbor_spacings(eigvals)
            spacings_histogram.add_histogram_contribution(nn_spacings)
            form_factors.compute_moment_contributions(eigvals)

            avg_unf_eigvals = self.ensemble.unfold_with_average_pdf(eigvals)

            spectral_histogram_avg_unfolded.add_histogram_contribution(avg_unf_eigvals)
            nn_spacings = self.compute_nearest_neighbor_spacings(avg_unf_eigvals)
            spacings_histogram_avg_unfolded.add_histogram_contribution(nn_spacings)
            form_factors_avg_unfolded.compute_moment_contributions(avg_unf_eigvals)

            var_unf_eigvals, _, _ = self.ensemble.unfold_with_variate_pdf(eigvals)

            spectral_histogram_var_unfolded.add_histogram_contribution(var_unf_eigvals)
            nn_spacings = self.compute_nearest_neighbor_spacings(var_unf_eigvals)
            spacings_histogram_var_unfolded.add_histogram_contribution(nn_spacings)
            form_factors_var_unfolded.compute_moment_contributions(var_unf_eigvals)
