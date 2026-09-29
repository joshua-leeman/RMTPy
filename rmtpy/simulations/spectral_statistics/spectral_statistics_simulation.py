from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import override

import attrs
from _typeshed import NoneType

from ...conversion import completed_at_utc
from ...ensembles.many_body_ensemble import ManyBodyEnsemble, RealEigenvalues
from ..base_data import Data
from ..base_simulation import Simulation
from ..histograms import Histogram
from ..statistics import (
    REALIZATIONS_METADATA,
    nearest_neighbor_spacings,
    truncated_polynomial_degree_range,
)
from ..unfolding import TruncatedPolynomialCDFFactory, unfold_values
from .nn_spacings_histogram import SpacingsHistogram
from .spectral_coefficients_histogram import SpectralCoefficientsHistogram
from .spectral_form_factors import FormFactorsData
from .spectral_histogram import SpectralHistogram


def run_spectral_statistics(
    ensemble: ManyBodyEnsemble,
    realizs: int,
) -> SpectralStatisticsSimulation:
    simulation = SpectralStatisticsSimulation(ensemble=ensemble, realizs=realizs)
    simulation.execute()
    return simulation


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class SpectralStatisticsBuffers:
    polynomial_degree: int | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(
            (attrs.validators.instance_of(int), attrs.validators.gt(0)),
        ),
    )

    levels: SpectralHistogram = attrs.field(
        validator=attrs.validators.instance_of(SpectralHistogram),
        repr=False,
    )
    nn_spacings: SpacingsHistogram = attrs.field(
        validator=attrs.validators.instance_of(SpacingsHistogram),
        repr=False,
    )
    form_factors: FormFactorsData = attrs.field(
        validator=attrs.validators.instance_of(FormFactorsData),
        repr=False,
    )

    def __iter__(self) -> Iterator[Data]:
        yield self.levels
        yield self.nn_spacings
        yield self.form_factors

    @classmethod
    def create_raw(
        cls,
        *,
        simulation: SpectralStatisticsSimulation,
    ) -> SpectralStatisticsBuffers:
        return SpectralStatisticsBuffers(
            levels=SpectralHistogram.create_raw(
                spectral_density=simulation.ensemble.spectral_density
            ),
            nn_spacings=SpacingsHistogram.create_raw(
                ensemble=simulation.ensemble,
            ),
            form_factors=FormFactorsData.create_raw(
                ensemble=simulation.ensemble,
                file_name_prefix="spectral_form_factors",
            ),
        )

    @classmethod
    def create_unfolded(
        cls,
        *,
        simulation: SpectralStatisticsSimulation,
        unfolding: str,
        polynomial_degree: int | None = None,
    ) -> SpectralStatisticsBuffers:
        suffix = f"_degree_{polynomial_degree}" if polynomial_degree is not None else ""

        return SpectralStatisticsBuffers(
            polynomial_degree=polynomial_degree,
            levels=SpectralHistogram.create_unfolded(
                file_name_prefix=f"spectral_histogram_{unfolding}_unfolded{suffix}",
                dimension=simulation.ensemble.dimension,
                unfolding=unfolding,
                polynomial_degree=polynomial_degree,
            ),
            nn_spacings=SpacingsHistogram.create_unfolded(
                file_name_prefix=f"spacings_histogram_{unfolding}_unfolded{suffix}",
                unfolding=unfolding,
                polynomial_degree=polynomial_degree,
            ),
            form_factors=FormFactorsData.create_unfolded(
                file_name_prefix=f"spectral_form_factors_{unfolding}_unfolded{suffix}",
                dimension=simulation.ensemble.dimension,
                unfolding=unfolding,
                polynomial_degree=polynomial_degree,
            ),
        )

    def accumulate_eigenvalues(
        self,
        eigvals: RealEigenvalues,
        *,
        degeneracy: int = 1,
    ) -> None:
        self.levels.add_histogram_contribution(eigvals)

        nn_spacings = nearest_neighbor_spacings(eigvals, degeneracy=degeneracy)
        self.nn_spacings.add_histogram_contribution(nn_spacings)

        self.form_factors.compute_moment_contributions(eigvals)


def _create_coefficient_buffers(
    simulation: SpectralStatisticsSimulation,
) -> Iterable[SpectralCoefficientsHistogram]:
    spectral_coeff_histogram_list: list[SpectralCoefficientsHistogram] = []
    for degree in range(1, simulation.ensemble.max_spectral_polynomial_degree + 1):
        spectral_coeff_histogram = SpectralCoefficientsHistogram.create(degree=degree)
        spectral_coeff_histogram_list.append(spectral_coeff_histogram)

    return tuple(spectral_coeff_histogram_list)


def _create_raw_buffers(
    simulation: SpectralStatisticsSimulation,
) -> SpectralStatisticsBuffers:
    return SpectralStatisticsBuffers.create_raw(simulation=simulation)


def _create_weight_unfolded_buffers(
    simulation: SpectralStatisticsSimulation,
) -> SpectralStatisticsBuffers:
    return SpectralStatisticsBuffers.create_unfolded(
        simulation=simulation,
        unfolding="weight",
    )


def _create_averaged_unfolded_buffers(
    simulation: SpectralStatisticsSimulation,
) -> tuple[SpectralStatisticsBuffers, ...]:
    polynomial_degrees = truncated_polynomial_degree_range(
        max_degree=simulation.ensemble.max_spectral_polynomial_degree
    )

    list_of_buffers: list[SpectralStatisticsBuffers] = []
    for polynomial_degree in polynomial_degrees:
        list_of_buffers.append(
            SpectralStatisticsBuffers.create_unfolded(
                simulation=simulation,
                unfolding="averaged",
                polynomial_degree=polynomial_degree,
            )
        )

    return tuple(list_of_buffers)


def _create_variate_unfolded_buffers(
    simulation: SpectralStatisticsSimulation,
) -> tuple[SpectralStatisticsBuffers, ...]:
    polynomial_degrees = truncated_polynomial_degree_range(
        max_degree=simulation.ensemble.max_spectral_polynomial_degree
    )

    list_of_buffers: list[SpectralStatisticsBuffers] = []
    for polynomial_degree in polynomial_degrees:
        list_of_buffers.append(
            SpectralStatisticsBuffers.create_unfolded(
                simulation=simulation,
                unfolding="variate",
                polynomial_degree=polynomial_degree,
            )
        )

    return tuple(list_of_buffers)


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class SpectralStatisticsSimulation(Simulation):
    ensemble: ManyBodyEnsemble = attrs.field(
        converter=ManyBodyEnsemble.create,
    )
    realizs: int = attrs.field(
        converter=int,
        validator=attrs.validators.gt(0),
        metadata=REALIZATIONS_METADATA,
    )

    coefficient_buffers: Iterable[SpectralCoefficientsHistogram] = attrs.field(
        default=attrs.Factory(_create_coefficient_buffers, takes_self=True),
        repr=False,
    )
    raw_buffers: SpectralStatisticsBuffers = attrs.field(
        default=attrs.Factory(_create_raw_buffers, takes_self=True),
        repr=False,
    )
    wgt_unfolded_buffers: SpectralStatisticsBuffers = attrs.field(
        default=attrs.Factory(_create_weight_unfolded_buffers, takes_self=True),
        repr=False,
    )
    ave_unfolded_buffers: Iterable[SpectralStatisticsBuffers] = attrs.field(
        default=attrs.Factory(_create_averaged_unfolded_buffers, takes_self=True)
    )
    var_unfolded_buffers: Iterable[SpectralStatisticsBuffers] = attrs.field(
        default=attrs.Factory(_create_variate_unfolded_buffers, takes_self=True)
    )

    def __iter__(self) -> Iterator[Data]:
        yield from self.coefficient_buffers
        yield from self.raw_buffers
        yield from self.wgt_unfolded_buffers

        for averaged_unfolded_buffers in self.ave_unfolded_buffers:
            yield from averaged_unfolded_buffers

        for variate_unfolded_buffers in self.var_unfolded_buffers:
            yield from variate_unfolded_buffers

    @property
    @override
    def _root_for_outputs(self) -> Path:
        return super()._root_for_outputs / self.ensemble.to_path

    def _build_cdf_factory(self) -> TruncatedPolynomialCDFFactory:
        return TruncatedPolynomialCDFFactory(
            density=self.ensemble.spectral_density,
            degrees=truncated_polynomial_degree_range(
                max_degree=self.ensemble.max_spectral_polynomial_degree
            ),
            density_name="spectral",
        )

    def _realize_spectral_statistics(self) -> None:
        degeneracy = self.ensemble.eigval_degeneracy
        dimension = self.ensemble.dimension
        density = self.ensemble.spectral_density

        cdf_factory = self._build_cdf_factory()

        average_cdfs = cdf_factory.average_interpolators()

        for eigvals in self.ensemble.eigvals_stream(realizs=self.realizs):
            self.raw_buffers.accumulate_eigenvalues(eigvals, degeneracy=degeneracy)

            variate_coeffs = density.compute_variate_coeffs(eigvals)
            for index, histogram in enumerate(self.coefficient_buffers, start=1):
                histogram.add_histogram_contribution(variate_coeffs[index : index + 1])

            wgt_unf_eigvals = unfold_values(
                eigvals, cdf=density.weight_cdf, dimension=dimension
            )
            self.wgt_unfolded_buffers.accumulate_eigenvalues(
                wgt_unf_eigvals, degeneracy=degeneracy
            )

            for cdf, buffers in zip(average_cdfs, self.ave_unfolded_buffers, strict=True):
                ave_unf_eigvals = unfold_values(eigvals, cdf=cdf, dimension=dimension)
                buffers.accumulate_eigenvalues(ave_unf_eigvals, degeneracy=degeneracy)

            variate_cdfs = cdf_factory.interpolators_from_coeffs(variate_coeffs)
            for cdf, buffers in zip(variate_cdfs, self.var_unfolded_buffers, strict=True):
                variate_levels = unfold_values(eigvals, cdf=cdf, dimension=dimension)
                buffers.accumulate_eigenvalues(variate_levels, degeneracy=degeneracy)

    def _finalize(self) -> None:
        for data in self:
            if isinstance(data, Histogram):
                data.compute_histogram()
            elif isinstance(data, FormFactorsData):
                data.compute_form_factors()

    @override
    def _execute(self) -> NoneType:
        spectral_density = self.ensemble.spectral_density
        has_average_coeffs = spectral_density.has_average_coeffs

        self._realize_spectral_statistics()
        self._finalize()

        calibration: dict[str, object] = {}
        if spectral_density.has_average_coeffs:
            calibration = {
                "density": "spectral",
                "average_coefficients": spectral_density.average_coeffs,
                "timing": (
                    "previously_cached"
                    if has_average_coeffs
                    else "cached_during_execution"
                ),
            }

        self._store_run_context(
            execution={
                "calibration": calibration,
                "completed_at_utc": completed_at_utc(),
            },
        )
