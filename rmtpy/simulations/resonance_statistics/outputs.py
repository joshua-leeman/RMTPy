from __future__ import annotations

from collections.abc import Iterator
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
    create_complex_energy_histogram_observable,
    create_resonance_coeff_histograms,
    create_resonance_form_factors_observable,
    create_resonance_histogram_observable,
    create_resonance_spacing_histogram_observable,
    create_unfolded_complex_energy_histogram_observable,
    create_unfolded_resonance_form_factors_observable,
    create_unfolded_resonance_histogram_observable,
    create_unfolded_resonance_spacing_histogram_observable,
    create_unfolded_width_histogram_observable,
    create_width_histogram_observable,
)
from .resonance_form_factors import FormFactorsData

if TYPE_CHECKING:
    from .resonance_statistics_simulation import ResonanceStatisticsSimulation


def create_unfolded_file_name(
    prefix: str,
    *,
    unfolding: str,
    degree: int | None,
) -> str:
    if degree is None:
        return f"{prefix}_wgt_unfolded"
    return f"{prefix}_{unfolding}_unfolded_deg_{degree}"


def create_unfolded_resonance_statistic_outputs(
    simulation: ResonanceStatisticsSimulation,
    *,
    unfolding: str,
    degree: int | None = None,
) -> ResonanceStatisticOutputs:
    if unfolding == "wgt":
        if degree is not None:
            raise ValueError("Weight unfolding does not use a polynomial degree.")
    elif unfolding in ("avg", "var"):
        if degree is None:
            raise ValueError(f"{unfolding} unfolding requires a polynomial degree.")
    else:
        raise ValueError(f"Unknown unfolding mode: {unfolding}.")

    return ResonanceStatisticOutputs(
        resonances=create_unfolded_resonance_histogram_observable(
            simulation,
            file_name=create_unfolded_file_name(
                "resonance_histogram",
                unfolding=unfolding,
                degree=degree,
            ),
            unfolding=unfolding,
            degree=degree,
        ),
        widths=create_unfolded_width_histogram_observable(
            file_name=create_unfolded_file_name(
                "width_histogram",
                unfolding=unfolding,
                degree=degree,
            ),
            unfolding=unfolding,
            degree=degree,
        ),
        spacings=create_unfolded_resonance_spacing_histogram_observable(
            file_name=create_unfolded_file_name(
                "resonance_spacing_histogram",
                unfolding=unfolding,
                degree=degree,
            ),
            unfolding=unfolding,
            degree=degree,
        ),
        complex_energies=create_unfolded_complex_energy_histogram_observable(
            file_name=create_unfolded_file_name(
                "complex_energy_histogram",
                unfolding=unfolding,
                degree=degree,
            ),
            unfolding=unfolding,
            degree=degree,
        ),
        form_factors=create_unfolded_resonance_form_factors_observable(
            simulation,
            file_name=create_unfolded_file_name(
                "resonance_form_factors",
                unfolding=unfolding,
                degree=degree,
            ),
            unfolding=unfolding,
            degree=degree,
        ),
    )


def create_resonance_statistics_outputs(
    simulation: ResonanceStatisticsSimulation,
) -> ResonanceStatisticsOutputs:
    truncated_degrees = simulation.truncated_degrees

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
        weight_unfolded=create_unfolded_resonance_statistic_outputs(
            simulation,
            unfolding="wgt",
        ),
        avg_unfolded_by_degree=tuple(
            create_unfolded_resonance_statistic_outputs(
                simulation,
                unfolding="avg",
                degree=degree,
            )
            for degree in truncated_degrees
        ),
        var_unfolded_by_degree=tuple(
            create_unfolded_resonance_statistic_outputs(
                simulation,
                unfolding="var",
                degree=degree,
            )
            for degree in truncated_degrees
        ),
    )


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class ResonanceStatisticOutputs:
    """The five resonance observables for one unfolding prescription."""

    resonances: Observable[Histogram]
    widths: Observable[Histogram]
    spacings: Observable[Histogram]
    complex_energies: Observable[Histogram2D]
    form_factors: Observable[FormFactorsData]

    def iter_observables(self) -> Iterator[Observable]:
        yield self.resonances
        yield self.widths
        yield self.spacings
        yield self.complex_energies
        yield self.form_factors

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
    """All resonance observables, grouped explicitly by unfolding and degree."""

    coefficients: CoefficientHistogramOutputs
    raw: ResonanceStatisticOutputs
    weight_unfolded: ResonanceStatisticOutputs
    avg_unfolded_by_degree: tuple[ResonanceStatisticOutputs, ...]
    var_unfolded_by_degree: tuple[ResonanceStatisticOutputs, ...]

    def iter_observables(self) -> Iterator[Observable]:
        yield from self.coefficients.iter_observables()
        yield from self.raw.iter_observables()
        yield from self.weight_unfolded.iter_observables()
        for outputs in self.avg_unfolded_by_degree:
            yield from outputs.iter_observables()
        for outputs in self.var_unfolded_by_degree:
            yield from outputs.iter_observables()
