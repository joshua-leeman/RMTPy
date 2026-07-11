from __future__ import annotations

from typing import TYPE_CHECKING

import attrs
import numpy as np

from rmtpy.ensembles import ManyBodyEnsemble

from ..histogram import Histogram
from ..histogram2D import Histogram2D
from ..observable import Observable
from ..outputs import CoefficientHistogramOutputs
from ..statistics import nearest_neighbor_spacings
from ..unfolding import CDF, unfold_values, unfold_widths
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
)
from .resonance_form_factors import FormFactorsData

if TYPE_CHECKING:
    from .resonance_statistics_simulation import ResonanceStatisticsSimulation


def _create_degree_outputs(
    resonances: list[Observable],
    widths: list[Observable],
    spacings: list[Observable],
    complex_energies: list[Observable],
    form_factors: list[Observable],
) -> tuple[ResonanceStatisticOutputs, ...]:
    return tuple(
        ResonanceStatisticOutputs(
            resonances=resonance,
            widths=width,
            spacings=spacing,
            complex_energies=complex_energy,
            form_factors=form_factor,
        )
        for resonance, width, spacing, complex_energy, form_factor in zip(
            resonances,
            widths,
            spacings,
            complex_energies,
            form_factors,
            strict=True,
        )
    )


def create_resonance_statistics_outputs(
    simulation: ResonanceStatisticsSimulation,
) -> ResonanceStatisticsOutputs:
    return ResonanceStatisticsOutputs(
        coefficients=CoefficientHistogramOutputs(
            by_degree=tuple(create_resonance_coeff_histograms(simulation))
        ),
        raw=ResonanceStatisticOutputs(
            resonances=create_resonance_histogram_observable(simulation),
            widths=create_width_histogram_observable(simulation),
            spacings=create_resonance_spacing_histogram_observable(simulation),
            complex_energies=create_complex_energy_histogram_observable(simulation),
            form_factors=create_resonance_form_factors_observable(simulation),
        ),
        weight_unfolded=ResonanceStatisticOutputs(
            resonances=create_weight_unfolded_resonance_histogram_observable(simulation),
            widths=create_weight_unfolded_width_histogram_observable(simulation),
            spacings=create_weight_unfolded_resonance_spacing_histogram_observable(
                simulation
            ),
            complex_energies=create_weight_unfolded_complex_energy_histogram_observable(
                simulation
            ),
            form_factors=create_weight_unfolded_resonance_form_factors_observable(
                simulation
            ),
        ),
        avg_unfolded_by_degree=_create_degree_outputs(
            create_avg_unfolded_resonance_histograms(simulation),
            create_avg_unfolded_width_histograms(simulation),
            create_avg_unfolded_resonance_spacing_histograms(simulation),
            create_avg_unfolded_complex_energy_histograms(simulation),
            create_avg_unfolded_resonance_form_factors(simulation),
        ),
        var_unfolded_by_degree=_create_degree_outputs(
            create_var_unfolded_resonance_histograms(simulation),
            create_var_unfolded_width_histograms(simulation),
            create_var_unfolded_resonance_spacing_histograms(simulation),
            create_var_unfolded_complex_energy_histograms(simulation),
            create_var_unfolded_resonance_form_factors(simulation),
        ),
    )


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class ResonanceStatisticOutputs:
    resonances: Observable[Histogram]
    widths: Observable[Histogram]
    spacings: Observable[Histogram]
    complex_energies: Observable[Histogram2D]
    form_factors: Observable[FormFactorsData]

    def add_raw(
        self,
        resonances: np.ndarray,
        widths: np.ndarray,
        *,
        ensemble: ManyBodyEnsemble,
    ) -> None:
        self.resonances.data.add_histogram_contribution(resonances)
        self.widths.data.add_histogram_contribution(widths / ensemble.spectral_radius)
        self.spacings.data.add_histogram_contribution(
            nearest_neighbor_spacings(
                resonances,
                degeneracy=ensemble.eigval_degeneracy,
            )
        )
        self.complex_energies.data.add_histogram_contribution(
            resonances / ensemble.spectral_radius,
            widths / ensemble.spectral_radius,
        )
        self.form_factors.data.compute_moment_contributions(resonances)

    def add_unfolded(
        self,
        resonances: np.ndarray,
        widths: np.ndarray,
        *,
        cdf: CDF,
        ensemble: ManyBodyEnsemble,
    ) -> None:
        unfolded_resonances = unfold_values(
            resonances,
            cdf=cdf,
            dimension=ensemble.dimension,
        )
        unfolded_widths = unfold_widths(
            widths,
            resonances,
            cdf=cdf,
            dimension=ensemble.dimension,
        )

        self.resonances.data.add_histogram_contribution(unfolded_resonances)
        self.widths.data.add_histogram_contribution(unfolded_widths)
        self.spacings.data.add_histogram_contribution(
            nearest_neighbor_spacings(
                unfolded_resonances,
                degeneracy=ensemble.eigval_degeneracy,
            )
        )
        self.complex_energies.data.add_histogram_contribution(
            resonances / ensemble.spectral_radius,
            unfolded_widths,
        )
        self.form_factors.data.compute_moment_contributions(unfolded_resonances)


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class ResonanceStatisticsOutputs:
    coefficients: CoefficientHistogramOutputs
    raw: ResonanceStatisticOutputs
    weight_unfolded: ResonanceStatisticOutputs
    avg_unfolded_by_degree: tuple[ResonanceStatisticOutputs, ...]
    var_unfolded_by_degree: tuple[ResonanceStatisticOutputs, ...]
