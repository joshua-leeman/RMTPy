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
from .spectral_coefficients import SpectralCoefficientHistogramPlot
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

REALIZATIONS_METADATA: dict[str, str] = {
    "dir_name": "realizs",
    "latex_name": "R",
}
TRUNCATED_SPECTRAL_POLYNOMIAL_DEGREE_MIN: int = 2
TRUNCATED_SPECTRAL_POLYNOMIAL_DEGREE_STEP: int = 2
LOG_D_TIME_SUPPORT: tuple[float, float] = (-0.5, 1.5)
LOG_D_UNFOLDED_TIME_SUPPORT: tuple[float, float] = (-1.5, 0.5)
NORMALIZED_SPACINGS_RANGE: np.ndarray = np.array([0.0, 4.0])
RANGE_SCALE_FACTOR_DEFAULT: float = 1.2
UNITLESS_DEFAULT_RANGE: np.ndarray = RANGE_SCALE_FACTOR_DEFAULT * np.array([-1.0, 1.0])


def thouless_time(times: np.ndarray, form_factor: np.ndarray) -> float:
    max_idx, _ = find_peaks(form_factor)

    pchip = PchipInterpolator(times[max_idx], form_factor[max_idx])

    start = max_idx[0]
    stop = max_idx[-1] + 1
    envelope = pchip(times[start:stop])
    relative_min_idx = np.argmin(envelope)
    thouless_idx = np.searchsorted(times, times[start:stop][relative_min_idx])

    return float(times[thouless_idx])


def iterate_truncated_spectral_polynomial_degrees(
    simulation: SpectralStatisticsSimulation,
) -> range:
    return range(
        TRUNCATED_SPECTRAL_POLYNOMIAL_DEGREE_MIN,
        simulation.ensemble.max_spectral_polynomial_degree + 1,
        TRUNCATED_SPECTRAL_POLYNOMIAL_DEGREE_STEP,
    )


def truncate_coeffs(coeffs: np.ndarray, truncate_degree: int) -> np.ndarray:
    truncated_coeffs: np.ndarray = np.zeros_like(coeffs)
    truncated_coeffs[: truncate_degree + 1] = coeffs[: truncate_degree + 1]
    return truncated_coeffs


def create_spectral_coeff_histograms(
    simulation: SpectralStatisticsSimulation,
) -> list[Observable]:
    histograms: list[Observable] = []
    for degree in range(1, simulation.ensemble.max_spectral_polynomial_degree + 1):
        histogram: Observable = Observable(
            data=Histogram(
                file_name=f"spectral_coeff_{degree}_histogram",
                support=(-0.2, 0.2),
                num_bins=50,
            ),
            plot_cls=SpectralCoefficientHistogramPlot,
            finalize=finalize_histogram,
        )

        histogram.metadata["degree"] = degree
        histograms.append(histogram)

    return histograms


def create_unfolded_spectral_histogram_observable(
    simulation: SpectralStatisticsSimulation,
    file_name: str,
    degree: int | None = None,
) -> Observable:
    histogram: Observable = Observable(
        data=Histogram(
            file_name=file_name,
            support=simulation.ensemble.dimension * UNITLESS_DEFAULT_RANGE,
        ),
        plot_cls=UnfoldedSpectralHistogramPlot,
        finalize=finalize_histogram,
    )
    if degree is not None:
        histogram.metadata["degree"] = degree

    return histogram


def create_unfolded_spacings_histogram_observable(
    file_name: str,
    degree: int | None = None,
) -> Observable:
    histogram: Observable = Observable(
        data=Histogram(
            file_name=file_name,
            support=NORMALIZED_SPACINGS_RANGE,
        ),
        plot_cls=UnfoldedSpacingsHistogramPlot,
        finalize=finalize_histogram,
    )
    if degree is not None:
        histogram.metadata["degree"] = degree

    return histogram


def create_unfolded_spectral_form_factors_observable(
    simulation: SpectralStatisticsSimulation,
    file_name: str,
    degree: int | None = None,
) -> Observable:
    form_factors: Observable = Observable(
        data=FormFactorsData(
            file_name=file_name,
            dimension=simulation.ensemble.dimension,
            logD_time_support=LOG_D_UNFOLDED_TIME_SUPPORT,
            scale=2 * np.pi,
        ),
        plot_cls=UnfoldedFormFactorsPlot,
        finalize=finalize_form_factors,
    )
    if degree is not None:
        form_factors.metadata["degree"] = degree

    return form_factors


def create_avg_unfolded_spectral_histograms(
    simulation: SpectralStatisticsSimulation,
) -> list[Observable]:
    return [
        create_unfolded_spectral_histogram_observable(
            simulation,
            file_name=f"spectral_histogram_avg_unfolded_degree_{degree}",
            degree=degree,
        )
        for degree in iterate_truncated_spectral_polynomial_degrees(simulation)
    ]


def create_var_unfolded_spectral_histograms(
    simulation: SpectralStatisticsSimulation,
) -> list[Observable]:
    return [
        create_unfolded_spectral_histogram_observable(
            simulation,
            file_name=f"spectral_histogram_var_unfolded_degree_{degree}",
            degree=degree,
        )
        for degree in iterate_truncated_spectral_polynomial_degrees(simulation)
    ]


def create_avg_unfolded_spacings_histograms(
    simulation: SpectralStatisticsSimulation,
) -> list[Observable]:
    return [
        create_unfolded_spacings_histogram_observable(
            file_name=f"spacings_histogram_avg_unfolded_degree_{degree}",
            degree=degree,
        )
        for degree in iterate_truncated_spectral_polynomial_degrees(simulation)
    ]


def create_var_unfolded_spacings_histograms(
    simulation: SpectralStatisticsSimulation,
) -> list[Observable]:
    return [
        create_unfolded_spacings_histogram_observable(
            file_name=f"spacings_histogram_var_unfolded_degree_{degree}",
            degree=degree,
        )
        for degree in iterate_truncated_spectral_polynomial_degrees(simulation)
    ]


def create_avg_unfolded_spectral_form_factors(
    simulation: SpectralStatisticsSimulation,
) -> list[Observable]:
    return [
        create_unfolded_spectral_form_factors_observable(
            simulation,
            file_name=f"spectral_form_factors_avg_unfolded_degree_{degree}",
            degree=degree,
        )
        for degree in iterate_truncated_spectral_polynomial_degrees(simulation)
    ]


def create_var_unfolded_spectral_form_factors(
    simulation: SpectralStatisticsSimulation,
) -> list[Observable]:
    return [
        create_unfolded_spectral_form_factors_observable(
            simulation,
            file_name=f"spectral_form_factors_var_unfolded_degree_{degree}",
            degree=degree,
        )
        for degree in iterate_truncated_spectral_polynomial_degrees(simulation)
    ]


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

    spectral_coeff_histograms: list[Observable] = attrs.field(
        default=attrs.Factory(create_spectral_coeff_histograms, takes_self=True),
        init=False,
        repr=False,
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

    spectral_histogram_wei_unfolded: Observable = attrs.field(repr=False)

    @spectral_histogram_wei_unfolded.default
    def create_wei_unfolded_spectral_histogram_observable(self) -> Observable:
        return Observable(
            data=Histogram(
                file_name="spectral_histogram_weight_unfolded",
                support=self.ensemble.dimension * UNITLESS_DEFAULT_RANGE,
            ),
            plot_cls=UnfoldedSpectralHistogramPlot,
            finalize=finalize_histogram,
        )

    spectral_histograms_avg_unfolded: list[Observable] = attrs.field(
        default=attrs.Factory(create_avg_unfolded_spectral_histograms, takes_self=True),
        init=False,
        repr=False,
    )

    spectral_histograms_var_unfolded: list[Observable] = attrs.field(
        default=attrs.Factory(create_var_unfolded_spectral_histograms, takes_self=True),
        init=False,
        repr=False,
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

    spacings_histogram_wei_unfolded: Observable = attrs.field(repr=False)

    @spacings_histogram_wei_unfolded.default
    def create_wei_unfolded_spacings_histogram_observable(self) -> Observable:
        return Observable(
            data=Histogram(
                file_name="spacings_histogram_wei_unfolded",
                support=NORMALIZED_SPACINGS_RANGE,
            ),
            plot_cls=UnfoldedSpacingsHistogramPlot,
            finalize=finalize_histogram,
        )

    spacings_histograms_avg_unfolded: list[Observable] = attrs.field(
        default=attrs.Factory(create_avg_unfolded_spacings_histograms, takes_self=True),
        init=False,
        repr=False,
    )

    spacings_histograms_var_unfolded: list[Observable] = attrs.field(
        default=attrs.Factory(create_var_unfolded_spacings_histograms, takes_self=True),
        init=False,
        repr=False,
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

    spectral_form_factors_wei_unfolded: Observable = attrs.field(repr=False)

    @spectral_form_factors_wei_unfolded.default
    def create_wei_unfolded_spectral_form_factors_observable(self) -> Observable:
        return Observable(
            data=FormFactorsData(
                file_name="spectral_form_factors_wei_unfolded",
                dimension=self.ensemble.dimension,
                logD_time_support=LOG_D_UNFOLDED_TIME_SUPPORT,
                scale=2 * np.pi,
            ),
            plot_cls=UnfoldedFormFactorsPlot,
            finalize=finalize_form_factors,
        )

    spectral_form_factors_avg_unfolded_by_degree: list[Observable] = attrs.field(
        default=attrs.Factory(
            create_avg_unfolded_spectral_form_factors,
            takes_self=True,
        ),
        init=False,
        repr=False,
    )

    spectral_form_factors_var_unfolded_by_degree: list[Observable] = attrs.field(
        default=attrs.Factory(
            create_var_unfolded_spectral_form_factors,
            takes_self=True,
        ),
        init=False,
        repr=False,
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

    def create_truncated_average_cdf_interpolators(self) -> list[PchipInterpolator]:
        average_coeffs: np.ndarray | None = (
            self.ensemble.spectral_density.average_coeffs
        )
        if average_coeffs is None:
            raise NotImplementedError(
                "Truncated polynomial unfolding requires a polynomial spectral "
                "density expansion."
            )

        return [
            self.ensemble.spectral_density.create_variate_cdf_interpolator(
                coeffs=truncate_coeffs(average_coeffs, degree)
            )
            for degree in iterate_truncated_spectral_polynomial_degrees(self)
        ]

    def realize_monte_carlo_simulation(self) -> None:
        truncated_degrees: tuple[int, ...] = tuple(
            iterate_truncated_spectral_polynomial_degrees(self)
        )
        avg_cdf_interpolators: list[PchipInterpolator] | None = None

        spec_coeff_hists: list[Observable] = self.spectral_coeff_histograms

        spec_hist: Histogram = self.spectral_histogram.data
        spec_hist_wei_unf: Histogram = self.spectral_histogram_wei_unfolded.data
        spec_hists_avg_unf: list[Histogram] = [
            observable.data for observable in self.spectral_histograms_avg_unfolded
        ]
        spec_hists_var_unf: list[Histogram] = [
            observable.data for observable in self.spectral_histograms_var_unfolded
        ]

        spac_hist: Histogram = self.spacings_histogram.data
        spac_hist_wei_unf: Histogram = self.spacings_histogram_wei_unfolded.data
        spac_hists_avg_unf: list[Histogram] = [
            observable.data for observable in self.spacings_histograms_avg_unfolded
        ]
        spac_hists_var_unf: list[Histogram] = [
            observable.data for observable in self.spacings_histograms_var_unfolded
        ]

        sff: FormFactorsData = self.spectral_form_factors.data
        sff_wei_unf: FormFactorsData = self.spectral_form_factors_wei_unfolded.data
        sffs_avg_unf: list[FormFactorsData] = [
            observable.data
            for observable in self.spectral_form_factors_avg_unfolded_by_degree
        ]
        sffs_var_unf: list[FormFactorsData] = [
            observable.data
            for observable in self.spectral_form_factors_var_unfolded_by_degree
        ]

        for eigvals in self.ensemble.eigvals_stream(self.realizs):
            spec_hist.add_histogram_contribution(eigvals)
            nn_spacings: np.ndarray = self.compute_nearest_neighbor_spacings(eigvals)
            spac_hist.add_histogram_contribution(nn_spacings)
            sff.compute_moment_contributions(eigvals)

            spec_coeffs: np.ndarray = (
                self.ensemble.spectral_density.compute_variate_coeffs(eigvals)
            )
            for list_idx, degree in enumerate(
                range(1, self.ensemble.max_spectral_polynomial_degree + 1)
            ):
                spec_coeff_hist: Histogram = spec_coeff_hists[list_idx].data
                spec_coeff_hist.add_histogram_contribution(spec_coeffs[degree])

            wei_unf_eigvals = rmtpy.density.unfold_with_cdf(
                eigvals,
                self.ensemble.spectral_density.weight_cdf,
                self.ensemble.dimension,
            )

            spec_hist_wei_unf.add_histogram_contribution(wei_unf_eigvals)
            nn_spacings = self.compute_nearest_neighbor_spacings(wei_unf_eigvals)
            spac_hist_wei_unf.add_histogram_contribution(nn_spacings)
            sff_wei_unf.compute_moment_contributions(wei_unf_eigvals)

            if avg_cdf_interpolators is None:
                avg_cdf_interpolators = (
                    self.create_truncated_average_cdf_interpolators()
                )

            for list_idx, avg_cdf_interpolator in enumerate(avg_cdf_interpolators):
                avg_unf_eigvals = rmtpy.density.unfold_with_cdf(
                    eigvals,
                    avg_cdf_interpolator,
                    self.ensemble.dimension,
                )

                spec_hists_avg_unf[list_idx].add_histogram_contribution(avg_unf_eigvals)
                nn_spacings = self.compute_nearest_neighbor_spacings(avg_unf_eigvals)
                spac_hists_avg_unf[list_idx].add_histogram_contribution(nn_spacings)
                sffs_avg_unf[list_idx].compute_moment_contributions(avg_unf_eigvals)

            for list_idx, degree in enumerate(truncated_degrees):
                var_cdf_interpolator: PchipInterpolator = (
                    self.ensemble.spectral_density.create_variate_cdf_interpolator(
                        coeffs=truncate_coeffs(spec_coeffs, degree)
                    )
                )
                var_unf_eigvals = rmtpy.density.unfold_with_cdf(
                    eigvals,
                    var_cdf_interpolator,
                    self.ensemble.dimension,
                )

                spec_hists_var_unf[list_idx].add_histogram_contribution(var_unf_eigvals)
                nn_spacings = self.compute_nearest_neighbor_spacings(var_unf_eigvals)
                spac_hists_var_unf[list_idx].add_histogram_contribution(nn_spacings)
                sffs_var_unf[list_idx].compute_moment_contributions(var_unf_eigvals)
