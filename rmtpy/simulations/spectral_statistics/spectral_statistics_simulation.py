from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import attrs
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks
from scipy.special import jn_zeros

import rmtpy.conversion
import rmtpy.density
import rmtpy.ensembles
from ..histogram import Histogram, finalize_histogram
from ..observable import Observable
from ..base import Simulation
from .spacings_histogram import (
    SpacingsHistogramPlot,
    UnfoldedSpacingsHistogramPlot,
)
from .spectral_form_factors import (
    finalize_form_factors,
    FormFactorsData,
    FormFactorsPlot,
    UnfoldedFormFactorsPlot,
)
from .spectral_histogram import (
    SpectralHistogramPlot,
    UnfoldedSpectralHistogramPlot,
)

DEFAULT_RANGE_SCALE_FACTOR: float = 1.2
LOG_D_TIME_SUPPORT: tuple[float, float] = (-0.5, 1.5)
LOG_D_UNFOLDED_TIME_SUPPORT: tuple[float, float] = (-1.5, 0.5)
REALIZATIONS_METADATA: dict[str, str] = {
    "dir_name": "realizs",
    "latex_name": "R",
}
NORMALIZED_SPACINGS_RANGE: np.ndarray = np.array([0.0, 4.0])
UNITLESS_DEFAULT_RANGE: np.ndarray = DEFAULT_RANGE_SCALE_FACTOR * np.array([-1.0, 1.0])


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
    SpectralStatisticsSimulation(ensemble=ensemble, realizs=realizs).run()


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class SpectralStatisticsSimulation(Simulation):
    ensemble: rmtpy.ensembles.ManyBodyEnsemble = attrs.field(
        converter=rmtpy.ensembles.ManyBodyEnsemble.create
    )
    realizs: int = attrs.field(
        converter=int,
        validator=attrs.validators.gt(0),
        metadata=REALIZATIONS_METADATA,
    )

    spectral_histogram: Observable = attrs.field(repr=False)

    @spectral_histogram.default
    def create_spectral_histogram_observable(self) -> Observable:
        return Observable(
            data=Histogram(
                file_name="spectral_histogram",
                support=self.ensemble.spectral_density.plot_range,
            ),
            plot_cls=SpectralHistogramPlot,
            finalize=finalize_histogram,
        )

    spectral_histogram_avg_unfolded: Observable = attrs.field(repr=False)

    @spectral_histogram_avg_unfolded.default
    def create_avg_unfolded_spectral_histogram_observable(self) -> Observable:
        return Observable(
            data=Histogram(
                file_name="spectral_histogram_avg_unfolded",
                support=self.ensemble.dimension * UNITLESS_DEFAULT_RANGE,
            ),
            plot_cls=UnfoldedSpectralHistogramPlot,
            finalize=finalize_histogram,
        )

    spectral_histogram_var_unfolded: Observable = attrs.field(repr=False)

    @spectral_histogram_var_unfolded.default
    def create_var_unfolded_spectral_histogram_observable(self) -> Observable:
        return Observable(
            data=Histogram(
                file_name="spectral_histogram_var_unfolded",
                support=self.ensemble.dimension * UNITLESS_DEFAULT_RANGE,
            ),
            plot_cls=UnfoldedSpectralHistogramPlot,
            finalize=finalize_histogram,
        )

    spacings_histogram: Observable = attrs.field(repr=False)

    @spacings_histogram.default
    def create_spacings_histogram_observable(self) -> Observable:
        global_mean_spacing: float = (
            2 * self.ensemble.spectral_radius / self.ensemble.dimension
        )

        spacings_histogram = Histogram(
            file_name="spacings_histogram",
            support=global_mean_spacing * NORMALIZED_SPACINGS_RANGE,
        )
        spacings_histogram.metadata["global_mean_spacing"] = global_mean_spacing

        return Observable(
            data=spacings_histogram,
            plot_cls=SpacingsHistogramPlot,
            finalize=finalize_histogram,
        )

    spacings_histogram_avg_unfolded: Observable = attrs.field(repr=False)

    @spacings_histogram_avg_unfolded.default
    def create_avg_unfolded_spacings_histogram_observable(self) -> Observable:
        return Observable(
            data=Histogram(
                file_name="spacings_histogram_avg_unfolded",
                support=NORMALIZED_SPACINGS_RANGE,
            ),
            plot_cls=UnfoldedSpacingsHistogramPlot,
            finalize=finalize_histogram,
        )

    spacings_histogram_var_unfolded: Observable = attrs.field(repr=False)

    @spacings_histogram_var_unfolded.default
    def create_var_unfolded_spacings_histogram_observable(self) -> Observable:
        return Observable(
            data=Histogram(
                file_name="spacings_histogram_var_unfolded",
                support=NORMALIZED_SPACINGS_RANGE,
            ),
            plot_cls=UnfoldedSpacingsHistogramPlot,
            finalize=finalize_histogram,
        )

    spectral_form_factors: Observable = attrs.field(repr=False)

    @spectral_form_factors.default
    def create_spectral_form_factors_observable(self) -> Observable:
        j_1_1: float = float(jn_zeros(1, 1)[0])
        return Observable(
            data=FormFactorsData(
                file_name="spectral_form_factors",
                dimension=self.ensemble.dimension,
                logD_time_support=LOG_D_TIME_SUPPORT,
                scale=j_1_1 / self.ensemble.spectral_radius,
            ),
            plot_cls=FormFactorsPlot,
            finalize=finalize_form_factors,
        )

    spectral_form_factors_avg_unfolded: Observable = attrs.field(repr=False)

    @spectral_form_factors_avg_unfolded.default
    def create_avg_unfolded_spectral_form_factors_observable(self) -> Observable:
        return Observable(
            data=FormFactorsData(
                file_name="spectral_form_factors_avg_unfolded",
                dimension=self.ensemble.dimension,
                logD_time_support=LOG_D_UNFOLDED_TIME_SUPPORT,
                scale=2 * np.pi,
            ),
            plot_cls=UnfoldedFormFactorsPlot,
            finalize=finalize_form_factors,
        )

    spectral_form_factors_var_unfolded: Observable = attrs.field(repr=False)

    @spectral_form_factors_var_unfolded.default
    def create_var_unfolded_spectral_form_factors_observable(self) -> Observable:
        return Observable(
            data=FormFactorsData(
                file_name="spectral_form_factors_var_unfolded",
                dimension=self.ensemble.dimension,
                logD_time_support=LOG_D_UNFOLDED_TIME_SUPPORT,
                scale=2 * np.pi,
            ),
            plot_cls=UnfoldedFormFactorsPlot,
            finalize=finalize_form_factors,
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
        self.metadata["args"]["ensemble"] = rmtpy.conversion.CONVERTER.unstructure(
            self.ensemble
        )
        self.metadata["args"]["realizs"] = self.realizs

    def compute_nearest_neighbor_spacings(self, eigvals: np.ndarray) -> np.ndarray:
        degeneracy: int = self.ensemble.eigval_degeneracy
        nn_spacings: np.ndarray = np.diff(np.sort(eigvals))
        if degeneracy > 1:
            nn_spacings = np.repeat(nn_spacings[1::degeneracy], degeneracy)
        return nn_spacings

    def realize_monte_carlo_simulation(self) -> None:
        spec_hist: Histogram = self.spectral_histogram.data
        spec_hist_avg_unf: Histogram = self.spectral_histogram_avg_unfolded.data
        spec_hist_var_unf: Histogram = self.spectral_histogram_var_unfolded.data

        spac_hist: Histogram = self.spacings_histogram.data
        spac_hist_avg_unf: Histogram = self.spacings_histogram_avg_unfolded.data
        spac_hist_var_unf: Histogram = self.spacings_histogram_var_unfolded.data

        sff: FormFactorsData = self.spectral_form_factors.data
        sff_avg_unf: FormFactorsData = self.spectral_form_factors_avg_unfolded.data
        sff_var_unf: FormFactorsData = self.spectral_form_factors_var_unfolded.data

        for eigvals in self.ensemble.eigvals_stream(self.realizs):
            print("eigvals =", np.sort(eigvals)[:8])
            spec_hist.add_histogram_contribution(eigvals)
            nn_spacings: np.ndarray = self.compute_nearest_neighbor_spacings(eigvals)
            spac_hist.add_histogram_contribution(nn_spacings)
            sff.compute_moment_contributions(eigvals)
            spec_coeffs: np.ndarray = (
                self.ensemble.spectral_density.compute_variate_coeffs(eigvals)
            )

            avg_unf_eigvals = rmtpy.density.unfold_with_cdf(
                eigvals,
                self.ensemble.spectral_density.average_cdf,
                self.ensemble.dimension,
            )

            spec_hist_avg_unf.add_histogram_contribution(avg_unf_eigvals)
            nn_spacings = self.compute_nearest_neighbor_spacings(avg_unf_eigvals)
            spac_hist_avg_unf.add_histogram_contribution(nn_spacings)
            sff_avg_unf.compute_moment_contributions(avg_unf_eigvals)

            var_cdf_interpolator: PchipInterpolator = (
                self.ensemble.spectral_density.create_variate_cdf_interpolator(
                    coeffs=spec_coeffs
                )
            )
            var_unf_eigvals = rmtpy.density.unfold_with_cdf(
                eigvals,
                var_cdf_interpolator,
                self.ensemble.dimension,
            )

            spec_hist_var_unf.add_histogram_contribution(var_unf_eigvals)
            nn_spacings = self.compute_nearest_neighbor_spacings(var_unf_eigvals)
            spac_hist_var_unf.add_histogram_contribution(nn_spacings)
            sff_var_unf.compute_moment_contributions(var_unf_eigvals)
