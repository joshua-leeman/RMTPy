from __future__ import annotations

from collections.abc import Iterator

import attrs
import numpy as np

from rmtpy.ensembles import ManyBodyEnsemble

from ..histogram import Histogram
from ..observable import Observable
from ..outputs import CoefficientHistogramOutputs
from ..statistics import nearest_neighbor_spacings, truncated_polynomial_degrees
from ..unfolding import CDF, unfold_values
from .observables import (
    create_raw_spacings_histogram_observable,
    create_raw_spectral_form_factors_observable,
    create_raw_spectral_histogram_observable,
    create_spectral_coeff_histograms,
    create_unfolded_spacings_histogram_observable,
    create_unfolded_spectral_form_factors_observable,
    create_unfolded_spectral_histogram_observable,
)
from .spectral_form_factors import FormFactorsData


def create_degree_unfolded_outputs(
    *,
    unfolding: str,
    degrees: tuple[int, ...],
    dimension: int,
) -> tuple[SpectralStatisticOutputs, ...]:
    return tuple(
        create_unfolded_outputs(
            unfolding=unfolding,
            level_file_name=f"spectral_histogram_{unfolding}_unfolded_degree_{degree}",
            spacing_file_name=(
                f"spacings_histogram_{unfolding}_unfolded_degree_{degree}"
            ),
            form_factor_file_name=(
                f"spectral_form_factors_{unfolding}_unfolded_degree_{degree}"
            ),
            dimension=dimension,
            degree=degree,
        )
        for degree in degrees
    )


def create_raw_outputs(ensemble: ManyBodyEnsemble) -> SpectralStatisticOutputs:
    return SpectralStatisticOutputs(
        levels=create_raw_spectral_histogram_observable(
            spectral_density=ensemble.spectral_density,
        ),
        spacings=create_raw_spacings_histogram_observable(ensemble=ensemble),
        form_factors=create_raw_spectral_form_factors_observable(ensemble=ensemble),
    )


def create_spectral_statistics_outputs(
    ensemble: ManyBodyEnsemble,
) -> SpectralStatisticsOutputs:
    degrees = tuple(truncated_polynomial_degrees(ensemble.max_spectral_polynomial_degree))

    return SpectralStatisticsOutputs(
        coefficients=CoefficientHistogramOutputs(
            by_degree=create_spectral_coeff_histograms(
                max_degree=ensemble.max_spectral_polynomial_degree,
            )
        ),
        raw=create_raw_outputs(ensemble),
        weight_unfolded=create_unfolded_outputs(
            unfolding="wgt",
            level_file_name="spectral_histogram_wgt_unfolded",
            spacing_file_name="spacings_histogram_wgt_unfolded",
            form_factor_file_name="spectral_form_factors_wgt_unfolded",
            dimension=ensemble.dimension,
        ),
        avg_unfolded_by_degree=create_degree_unfolded_outputs(
            unfolding="avg",
            degrees=degrees,
            dimension=ensemble.dimension,
        ),
        var_unfolded_by_degree=create_degree_unfolded_outputs(
            unfolding="var",
            degrees=degrees,
            dimension=ensemble.dimension,
        ),
    )


def create_unfolded_outputs(
    *,
    unfolding: str,
    level_file_name: str,
    spacing_file_name: str,
    form_factor_file_name: str,
    dimension: int,
    degree: int | None = None,
) -> SpectralStatisticOutputs:
    return SpectralStatisticOutputs(
        levels=create_unfolded_spectral_histogram_observable(
            file_name=level_file_name,
            dimension=dimension,
            unfolding=unfolding,
            degree=degree,
        ),
        spacings=create_unfolded_spacings_histogram_observable(
            file_name=spacing_file_name,
            unfolding=unfolding,
            degree=degree,
        ),
        form_factors=create_unfolded_spectral_form_factors_observable(
            file_name=form_factor_file_name,
            dimension=dimension,
            unfolding=unfolding,
            degree=degree,
        ),
    )


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class SpectralStatisticOutputs:
    """The three spectral observables for one unfolding prescription."""

    levels: Observable[Histogram]
    spacings: Observable[Histogram]
    form_factors: Observable[FormFactorsData]

    def iter_observables(self) -> Iterator[Observable]:
        yield self.levels
        yield self.spacings
        yield self.form_factors

    def add_levels(self, levels: np.ndarray, *, degeneracy: int = 1) -> None:
        spacings = nearest_neighbor_spacings(
            levels,
            degeneracy=degeneracy,
        )
        self.levels.data.add_histogram_contribution(levels)
        self.spacings.data.add_histogram_contribution(spacings)
        self.form_factors.data.compute_moment_contributions(levels)

    def add_unfolded_levels(
        self,
        levels: np.ndarray,
        *,
        degeneracy: int = 1,
        cdf: CDF,
        dimension: int,
    ) -> None:
        self.add_levels(
            unfold_values(levels, cdf=cdf, dimension=dimension), degeneracy=degeneracy
        )


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class SpectralStatisticsOutputs:
    """All spectral observables, grouped explicitly by unfolding and degree."""

    coefficients: CoefficientHistogramOutputs
    raw: SpectralStatisticOutputs
    weight_unfolded: SpectralStatisticOutputs
    avg_unfolded_by_degree: tuple[SpectralStatisticOutputs, ...]
    var_unfolded_by_degree: tuple[SpectralStatisticOutputs, ...]

    def iter_observables(self) -> Iterator[Observable]:
        yield from self.coefficients.iter_observables()
        yield from self.raw.iter_observables()
        yield from self.weight_unfolded.iter_observables()

        for outputs in self.avg_unfolded_by_degree:
            yield from outputs.iter_observables()

        for outputs in self.var_unfolded_by_degree:
            yield from outputs.iter_observables()
