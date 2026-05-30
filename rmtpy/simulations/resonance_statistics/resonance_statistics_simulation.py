from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import attrs
import numpy as np
from scipy.interpolate import PchipInterpolator

import rmtpy.density
from rmtpy.compounds import Compound
from rmtpy.conversion import RMT_CONVERTER
from rmtpy.ensembles import ManyBodyEnsemble

from ..base import Simulation
from ..histogram import Histogram
from ..histogram2D import Histogram2D
from ..observable import Observable
from ..statistics import (
    create_truncated_cdf_interpolators,
    nearest_neighbor_spacings,
    observable_data,
    observable_data_list,
    simulation_output_path,
    truncate_coeffs,
)
from .observables import (
    create_avg_unfolded_complex_energy_histograms,
    create_avg_unfolded_resonance_form_factors,
    create_avg_unfolded_resonance_histograms,
    create_avg_unfolded_resonance_spacing_histograms,
    create_avg_unfolded_width_histograms,
    create_complex_energy_histogram_observable,
    create_resonance_coeff_histograms,
    create_resonance_form_factors_observable,
    create_resonance_histogram_observable,
    create_resonance_spacing_histogram_observable,
    create_var_unfolded_complex_energy_histograms,
    create_var_unfolded_resonance_form_factors,
    create_var_unfolded_resonance_histograms,
    create_var_unfolded_resonance_spacing_histograms,
    create_var_unfolded_width_histograms,
    create_weight_unfolded_complex_energy_histogram_observable,
    create_weight_unfolded_resonance_form_factors_observable,
    create_weight_unfolded_resonance_histogram_observable,
    create_weight_unfolded_resonance_spacing_histogram_observable,
    create_weight_unfolded_width_histogram_observable,
    create_width_histogram_observable,
    iterate_truncated_spectral_polynomial_degrees,
)
from .resonance_form_factors import FormFactorsData

REALIZATIONS_METADATA: dict[str, str] = {
    "dir_name": "realizs",
    "latex_name": "R",
}


def run_resonance_statistics(compound: Compound, realizs: int) -> None:
    ResonanceStatisticsSimulation(compound=compound, realizs=realizs).run()


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class ResonanceStatisticTargets:
    resonances: Histogram
    widths: Histogram
    spacings: Histogram
    complex_energies: Histogram2D
    form_factors: FormFactorsData


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class ResonanceStatisticsSimulation(Simulation):
    compound: Compound = attrs.field(
        converter=Compound.create,
    )
    realizs: int = attrs.field(
        converter=int,
        validator=attrs.validators.gt(0),
        metadata=REALIZATIONS_METADATA,
    )

    resonance_coeff_histograms: list[Observable] = attrs.field(
        default=attrs.Factory(create_resonance_coeff_histograms, takes_self=True),
        init=False,
        repr=False,
    )

    resonance_histogram: Observable = attrs.field(
        default=attrs.Factory(create_resonance_histogram_observable, takes_self=True),
        repr=False,
    )

    resonance_histogram_wgt_unfolded: Observable = attrs.field(
        default=attrs.Factory(
            create_weight_unfolded_resonance_histogram_observable,
            takes_self=True,
        ),
        repr=False,
    )

    resonance_histograms_avg_unfolded: list[Observable] = attrs.field(
        default=attrs.Factory(
            create_avg_unfolded_resonance_histograms,
            takes_self=True,
        ),
        init=False,
        repr=False,
    )

    resonance_histograms_var_unfolded: list[Observable] = attrs.field(
        default=attrs.Factory(
            create_var_unfolded_resonance_histograms,
            takes_self=True,
        ),
        init=False,
        repr=False,
    )

    width_histogram: Observable = attrs.field(
        default=attrs.Factory(create_width_histogram_observable, takes_self=True),
        repr=False,
    )

    width_histogram_wgt_unfolded: Observable = attrs.field(
        default=attrs.Factory(
            create_weight_unfolded_width_histogram_observable,
            takes_self=True,
        ),
        repr=False,
    )

    width_histograms_avg_unfolded: list[Observable] = attrs.field(
        default=attrs.Factory(create_avg_unfolded_width_histograms, takes_self=True),
        init=False,
        repr=False,
    )

    width_histograms_var_unfolded: list[Observable] = attrs.field(
        default=attrs.Factory(create_var_unfolded_width_histograms, takes_self=True),
        init=False,
        repr=False,
    )

    resonance_spacing_histogram: Observable = attrs.field(
        default=attrs.Factory(
            create_resonance_spacing_histogram_observable,
            takes_self=True,
        ),
        repr=False,
    )

    resonance_spacing_histogram_wgt_unfolded: Observable = attrs.field(
        default=attrs.Factory(
            create_weight_unfolded_resonance_spacing_histogram_observable,
            takes_self=True,
        ),
        repr=False,
    )

    resonance_spacing_histograms_avg_unfolded: list[Observable] = attrs.field(
        default=attrs.Factory(
            create_avg_unfolded_resonance_spacing_histograms,
            takes_self=True,
        ),
        init=False,
        repr=False,
    )

    resonance_spacing_histograms_var_unfolded: list[Observable] = attrs.field(
        default=attrs.Factory(
            create_var_unfolded_resonance_spacing_histograms,
            takes_self=True,
        ),
        init=False,
        repr=False,
    )

    complex_energy_histogram: Observable = attrs.field(
        default=attrs.Factory(
            create_complex_energy_histogram_observable, takes_self=True
        ),
        repr=False,
    )

    complex_energy_histogram_wgt_unfolded: Observable = attrs.field(
        default=attrs.Factory(
            create_weight_unfolded_complex_energy_histogram_observable,
            takes_self=True,
        ),
        repr=False,
    )

    complex_energy_histograms_avg_unfolded: list[Observable] = attrs.field(
        default=attrs.Factory(
            create_avg_unfolded_complex_energy_histograms,
            takes_self=True,
        ),
        init=False,
        repr=False,
    )

    complex_energy_histograms_var_unfolded: list[Observable] = attrs.field(
        default=attrs.Factory(
            create_var_unfolded_complex_energy_histograms,
            takes_self=True,
        ),
        init=False,
        repr=False,
    )

    resonance_form_factors: Observable = attrs.field(
        default=attrs.Factory(
            create_resonance_form_factors_observable, takes_self=True
        ),
        repr=False,
    )

    resonance_form_factors_wgt_unfolded: Observable = attrs.field(
        default=attrs.Factory(
            create_weight_unfolded_resonance_form_factors_observable,
            takes_self=True,
        ),
        repr=False,
    )

    resonance_form_factors_avg_unfolded_by_degree: list[Observable] = attrs.field(
        default=attrs.Factory(
            create_avg_unfolded_resonance_form_factors,
            takes_self=True,
        ),
        init=False,
        repr=False,
    )

    resonance_form_factors_var_unfolded_by_degree: list[Observable] = attrs.field(
        default=attrs.Factory(
            create_var_unfolded_resonance_form_factors,
            takes_self=True,
        ),
        init=False,
        repr=False,
    )

    @property
    def to_path(self) -> Path:
        return simulation_output_path(
            self,
            Path(self.path_name) / self.compound.to_path,
        )

    def populate_metadata(self) -> None:
        super().populate_metadata()
        self.metadata["args"]["compound"] = RMT_CONVERTER.unstructure(self.compound)
        self.metadata["args"]["realizs"] = self.realizs

    def compute_nearest_neighbor_spacings(self, resonances: np.ndarray) -> np.ndarray:
        return nearest_neighbor_spacings(
            resonances,
            self.compound.ensemble.eigval_degeneracy,
        )

    def create_truncated_average_cdf_interpolators(self) -> list[PchipInterpolator]:
        return create_truncated_cdf_interpolators(
            self.compound.resonance_density,
            self.truncated_degrees,
            density_name="resonance",
        )

    @property
    def truncated_degrees(self) -> tuple[int, ...]:
        return tuple(iterate_truncated_spectral_polynomial_degrees(self))

    def create_targets(
        self,
        resonances: Observable,
        widths: Observable,
        spacings: Observable,
        complex_energies: Observable,
        form_factors: Observable,
    ) -> ResonanceStatisticTargets:
        return ResonanceStatisticTargets(
            resonances=observable_data(resonances, Histogram),
            widths=observable_data(widths, Histogram),
            spacings=observable_data(spacings, Histogram),
            complex_energies=observable_data(complex_energies, Histogram2D),
            form_factors=observable_data(form_factors, FormFactorsData),
        )

    def create_targets_by_degree(
        self,
        resonance_observables: list[Observable],
        width_observables: list[Observable],
        spacing_observables: list[Observable],
        complex_energy_observables: list[Observable],
        form_factor_observables: list[Observable],
    ) -> list[ResonanceStatisticTargets]:
        return [
            self.create_targets(
                resonances, widths, spacings, complex_energies, form_factors
            )
            for (resonances, widths, spacings, complex_energies, form_factors) in zip(
                resonance_observables,
                width_observables,
                spacing_observables,
                complex_energy_observables,
                form_factor_observables,
                strict=True,
            )
        ]

    def add_raw_contributions(
        self,
        resonances: np.ndarray,
        widths: np.ndarray,
        targets: ResonanceStatisticTargets,
    ) -> None:
        energy_0: float = self.compound.ensemble.spectral_radius

        targets.resonances.add_histogram_contribution(resonances)
        targets.widths.add_histogram_contribution(widths / energy_0)
        targets.spacings.add_histogram_contribution(
            self.compute_nearest_neighbor_spacings(resonances)
        )
        targets.complex_energies.add_histogram_contribution(
            resonances / energy_0,
            widths / energy_0,
        )
        targets.form_factors.compute_moment_contributions(resonances)

    def add_coefficient_contributions(
        self,
        resonances: np.ndarray,
        histograms: list[Histogram],
    ) -> np.ndarray:
        coeffs = self.compound.resonance_density.compute_variate_coeffs(resonances)
        max_degree: int = self.compound.ensemble.max_spectral_polynomial_degree
        for histogram, degree in zip(histograms, range(1, max_degree + 1), strict=True):
            histogram.add_histogram_contribution(coeffs[degree])
        return coeffs

    def add_unfolded_contributions(
        self,
        resonances: np.ndarray,
        widths: np.ndarray,
        cdf: Callable[[np.ndarray], np.ndarray],
        targets: ResonanceStatisticTargets,
    ) -> None:
        ensemble: ManyBodyEnsemble = self.compound.ensemble
        dimension: int = ensemble.dimension
        energy_0: float = ensemble.spectral_radius

        unfolded_resonances: np.ndarray = rmtpy.density.unfold_with_cdf(
            resonances, cdf, dimension
        )
        unfolded_widths: np.ndarray = rmtpy.density.unfold_widths_with_cdf(
            widths, resonances, cdf, dimension
        )

        targets.resonances.add_histogram_contribution(unfolded_resonances)
        targets.widths.add_histogram_contribution(unfolded_widths)

        nn_spacings: np.ndarray = self.compute_nearest_neighbor_spacings(
            unfolded_resonances
        )
        targets.spacings.add_histogram_contribution(nn_spacings)

        targets.complex_energies.add_histogram_contribution(
            resonances / energy_0, unfolded_widths
        )

        targets.form_factors.compute_moment_contributions(unfolded_resonances)

    def realize_monte_carlo_simulation(self) -> None:
        resonance_density: rmtpy.density.DensityModel = self.compound.resonance_density
        truncated_degrees: tuple[int, ...] = self.truncated_degrees
        avg_cdf_interpolators: list[PchipInterpolator] | None = None

        compound: Compound = self.compound
        coefficient_histograms = observable_data_list(
            self.resonance_coeff_histograms,
            Histogram,
        )
        raw_targets = self.create_targets(
            self.resonance_histogram,
            self.width_histogram,
            self.resonance_spacing_histogram,
            self.complex_energy_histogram,
            self.resonance_form_factors,
        )
        weight_targets = self.create_targets(
            self.resonance_histogram_wgt_unfolded,
            self.width_histogram_wgt_unfolded,
            self.resonance_spacing_histogram_wgt_unfolded,
            self.complex_energy_histogram_wgt_unfolded,
            self.resonance_form_factors_wgt_unfolded,
        )
        average_targets = self.create_targets_by_degree(
            self.resonance_histograms_avg_unfolded,
            self.width_histograms_avg_unfolded,
            self.resonance_spacing_histograms_avg_unfolded,
            self.complex_energy_histograms_avg_unfolded,
            self.resonance_form_factors_avg_unfolded_by_degree,
        )
        variate_targets = self.create_targets_by_degree(
            self.resonance_histograms_var_unfolded,
            self.width_histograms_var_unfolded,
            self.resonance_spacing_histograms_var_unfolded,
            self.complex_energy_histograms_var_unfolded,
            self.resonance_form_factors_var_unfolded_by_degree,
        )

        for complex_energies in compound.resonances_stream(self.realizs):
            resonances: np.ndarray = complex_energies.real
            widths: np.ndarray = -2 * complex_energies.imag

            self.add_raw_contributions(resonances, widths, raw_targets)
            resonance_coeffs = self.add_coefficient_contributions(
                resonances,
                coefficient_histograms,
            )

            self.add_unfolded_contributions(
                resonances,
                widths,
                resonance_density.weight_cdf,
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
                    resonances,
                    widths,
                    avg_cdf_interpolator,
                    targets,
                )

            for degree, targets in zip(
                truncated_degrees,
                variate_targets,
                strict=True,
            ):
                var_cdf_interpolator: PchipInterpolator = (
                    resonance_density.create_variate_cdf_interpolator(
                        coeffs=truncate_coeffs(resonance_coeffs, degree)
                    )
                )
                self.add_unfolded_contributions(
                    resonances,
                    widths,
                    var_cdf_interpolator,
                    targets,
                )
