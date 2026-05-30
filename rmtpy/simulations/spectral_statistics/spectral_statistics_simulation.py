from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import attrs
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks

import rmtpy.density
from rmtpy.conversion import RMT_CONVERTER
from rmtpy.ensembles import ManyBodyEnsemble

from ..base import Simulation
from ..histogram import Histogram
from ..observable import Observable
from ..statistics import (
    REALIZATIONS_METADATA,
    create_truncated_average_cdf_interpolators,
    nearest_neighbor_spacings,
    observable_data,
    observable_data_list,
    simulation_output_path,
    truncate_coeffs,
)
from .observables import (
    create_avg_unfolded_spacings_histograms,
    create_avg_unfolded_spectral_form_factors,
    create_avg_unfolded_spectral_histograms,
    create_spacings_histogram_observable,
    create_spectral_coeff_histograms,
    create_spectral_form_factors_observable,
    create_spectral_histogram_observable,
    create_var_unfolded_spacings_histograms,
    create_var_unfolded_spectral_form_factors,
    create_var_unfolded_spectral_histograms,
    create_weight_unfolded_spacings_histogram_observable,
    create_weight_unfolded_spectral_form_factors_observable,
    create_weight_unfolded_spectral_histogram_observable,
    iterate_truncated_spectral_polynomial_degrees,
)
from .spectral_form_factors import FormFactorsData


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
    SpectralStatisticsSimulation(ensemble=ensemble, realizs=realizs).run()


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class SpectralStatisticTargets:
    levels: Histogram
    spacings: Histogram
    form_factors: FormFactorsData


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class SpectralStatisticsSimulation(Simulation):
    ensemble: ManyBodyEnsemble = attrs.field(
        converter=ManyBodyEnsemble.create,
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

    spectral_histogram: Observable = attrs.field(
        default=attrs.Factory(create_spectral_histogram_observable, takes_self=True),
        repr=False,
    )
    spectral_histogram_wgt_unfolded: Observable = attrs.field(
        default=attrs.Factory(
            create_weight_unfolded_spectral_histogram_observable,
            takes_self=True,
        ),
        repr=False,
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

    spacings_histogram: Observable = attrs.field(
        default=attrs.Factory(create_spacings_histogram_observable, takes_self=True),
        repr=False,
    )
    spacings_histogram_wgt_unfolded: Observable = attrs.field(
        default=attrs.Factory(
            create_weight_unfolded_spacings_histogram_observable,
            takes_self=True,
        ),
        repr=False,
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

    spectral_form_factors: Observable = attrs.field(
        default=attrs.Factory(create_spectral_form_factors_observable, takes_self=True),
        repr=False,
    )
    spectral_form_factors_wgt_unfolded: Observable = attrs.field(
        default=attrs.Factory(
            create_weight_unfolded_spectral_form_factors_observable,
            takes_self=True,
        ),
        repr=False,
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
        return simulation_output_path(
            self,
            Path(self.path_name) / self.ensemble.to_path,
        )

    @property
    def truncated_degrees(self) -> tuple[int, ...]:
        return tuple(iterate_truncated_spectral_polynomial_degrees(self))

    def populate_metadata(self) -> None:
        super().populate_metadata()
        self.metadata["args"]["ensemble"] = RMT_CONVERTER.unstructure(self.ensemble)
        self.metadata["args"]["realizs"] = self.realizs

    def compute_nearest_neighbor_spacings(self, eigvals: np.ndarray) -> np.ndarray:
        return nearest_neighbor_spacings(eigvals, self.ensemble.eigval_degeneracy)

    def create_truncated_average_cdf_interpolators(self) -> list[PchipInterpolator]:
        return create_truncated_average_cdf_interpolators(
            self.ensemble.spectral_density,
            self.truncated_degrees,
            density_name="spectral",
        )

    def create_targets(
        self,
        levels: Observable,
        spacings: Observable,
        form_factors: Observable,
    ) -> SpectralStatisticTargets:
        return SpectralStatisticTargets(
            levels=observable_data(levels, Histogram),
            spacings=observable_data(spacings, Histogram),
            form_factors=observable_data(form_factors, FormFactorsData),
        )

    def create_targets_by_degree(
        self,
        level_observables: list[Observable],
        spacing_observables: list[Observable],
        form_factor_observables: list[Observable],
    ) -> list[SpectralStatisticTargets]:
        return [
            self.create_targets(levels, spacings, form_factors)
            for levels, spacings, form_factors in zip(
                level_observables,
                spacing_observables,
                form_factor_observables,
                strict=True,
            )
        ]

    def add_raw_contributions(
        self,
        eigvals: np.ndarray,
        targets: SpectralStatisticTargets,
    ) -> None:
        targets.levels.add_histogram_contribution(eigvals)
        targets.spacings.add_histogram_contribution(
            self.compute_nearest_neighbor_spacings(eigvals)
        )
        targets.form_factors.compute_moment_contributions(eigvals)

    def add_coefficient_contributions(
        self,
        eigvals: np.ndarray,
        histograms: list[Histogram],
    ) -> np.ndarray:
        coeffs: np.ndarray = self.ensemble.spectral_density.compute_variate_coeffs(
            eigvals
        )
        for histogram, degree in zip(
            histograms,
            range(1, self.ensemble.max_spectral_polynomial_degree + 1),
            strict=True,
        ):
            histogram.add_histogram_contribution(coeffs[degree])
        return coeffs

    def add_unfolded_contributions(
        self,
        eigvals: np.ndarray,
        cdf: Callable[[np.ndarray], np.ndarray],
        targets: SpectralStatisticTargets,
    ) -> None:
        unfolded_eigvals: np.ndarray = rmtpy.density.unfold_with_cdf(
            eigvals,
            cdf=cdf,
            dimensnion=self.ensemble.dimension,
        )

        unfolded_nn_spacings: np.ndarray = nearest_neighbor_spacings(unfolded_eigvals)

        targets.levels.add_histogram_contribution(unfolded_eigvals)
        targets.spacings.add_histogram_contribution(unfolded_nn_spacings)
        targets.form_factors.compute_moment_contributions(unfolded_eigvals)

    def realize_monte_carlo_simulation(self) -> None:
        avg_cdf_interpolators: list[PchipInterpolator] | None = None

        coefficient_histograms = observable_data_list(
            self.spectral_coeff_histograms,
            Histogram,
        )
        raw_targets = self.create_targets(
            self.spectral_histogram,
            self.spacings_histogram,
            self.spectral_form_factors,
        )
        weight_targets = self.create_targets(
            self.spectral_histogram_wgt_unfolded,
            self.spacings_histogram_wgt_unfolded,
            self.spectral_form_factors_wgt_unfolded,
        )
        average_targets = self.create_targets_by_degree(
            self.spectral_histograms_avg_unfolded,
            self.spacings_histograms_avg_unfolded,
            self.spectral_form_factors_avg_unfolded_by_degree,
        )
        variate_targets = self.create_targets_by_degree(
            self.spectral_histograms_var_unfolded,
            self.spacings_histograms_var_unfolded,
            self.spectral_form_factors_var_unfolded_by_degree,
        )

        for eigvals in self.ensemble.eigvals_stream(self.realizs):
            self.add_raw_contributions(eigvals, raw_targets)
            spec_coeffs = self.add_coefficient_contributions(
                eigvals,
                coefficient_histograms,
            )

            self.add_unfolded_contributions(
                eigvals,
                self.ensemble.spectral_density.weight_cdf,
                weight_targets,
            )

            if avg_cdf_interpolators is None:
                avg_cdf_interpolators = (
                    self.create_truncated_average_cdf_interpolators()
                )

            for avg_cdf_interpolator, targets in zip(
                avg_cdf_interpolators,
                average_targets,
                strict=True,
            ):
                self.add_unfolded_contributions(
                    eigvals,
                    avg_cdf_interpolator,
                    targets,
                )

            for degree, targets in zip(
                self.truncated_degrees,
                variate_targets,
                strict=True,
            ):
                var_cdf_interpolator: PchipInterpolator = (
                    self.ensemble.spectral_density.create_variate_cdf_interpolator(
                        coeffs=truncate_coeffs(spec_coeffs, degree)
                    )
                )
                self.add_unfolded_contributions(
                    eigvals,
                    var_cdf_interpolator,
                    targets,
                )
