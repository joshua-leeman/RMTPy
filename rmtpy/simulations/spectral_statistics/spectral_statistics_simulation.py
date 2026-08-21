from __future__ import annotations

from pathlib import Path

import attrs

import rmtpy.conversion
from rmtpy.ensembles import ManyBodyEnsemble

from ..base import Simulation
from ..statistics import REALIZATIONS_METADATA, truncated_polynomial_degrees
from ..unfolding import TruncatedPolynomialCdfFactory
from .outputs import (
    SpectralStatisticsOutputs,
    create_spectral_statistics_outputs,
)


def create_outputs(
    simulation: SpectralStatisticsSimulation,
) -> SpectralStatisticsOutputs:
    return create_spectral_statistics_outputs(simulation.ensemble)


def run_spectral_statistics(ensemble: ManyBodyEnsemble, realizs: int) -> None:
    SpectralStatisticsSimulation(ensemble=ensemble, realizs=realizs).run()


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class SpectralStatisticsSimulation(Simulation):
    """Monte Carlo experiment for closed-system spectral statistics."""

    ensemble: ManyBodyEnsemble = attrs.field(
        converter=ManyBodyEnsemble.create,
    )
    realizs: int = attrs.field(
        converter=int,
        validator=attrs.validators.gt(0),
        metadata=REALIZATIONS_METADATA,
    )

    outputs: SpectralStatisticsOutputs = attrs.field(
        default=attrs.Factory(create_outputs, takes_self=True),
        init=False,
        repr=False,
    )

    @property
    def to_path(self) -> Path:
        return rmtpy.conversion.to_path(
            self,
            root=Path(self.path_name) / self.ensemble.to_path,
        )

    @property
    def truncated_degrees(self) -> tuple[int, ...]:
        return tuple(
            truncated_polynomial_degrees(self.ensemble.max_spectral_polynomial_degree)
        )

    def create_cdf_factory(self) -> TruncatedPolynomialCdfFactory:
        return TruncatedPolynomialCdfFactory(
            density=self.ensemble.spectral_density,
            degrees=self.truncated_degrees,
            density_name="spectral",
        )

    def realize_monte_carlo_simulation(self) -> None:
        cdf_factory = self.create_cdf_factory()
        avg_cdf_interpolators = cdf_factory.average_interpolators()

        for eigvals in self.ensemble.eigvals_stream(realizs=self.realizs):
            self.outputs.raw.add_levels(
                eigvals,
                degeneracy=self.ensemble.eigval_degeneracy,
            )
            coeffs = self.outputs.coefficients.add(
                eigvals,
                density=self.ensemble.spectral_density,
            )

            self.outputs.weight_unfolded.add_unfolded_levels(
                eigvals,
                degeneracy=self.ensemble.eigval_degeneracy,
                cdf=self.ensemble.spectral_density.weight_cdf,
                dimension=self.ensemble.dimension,
            )

            for cdf, outputs in zip(
                avg_cdf_interpolators,
                self.outputs.avg_unfolded_by_degree,
                strict=True,
            ):
                outputs.add_unfolded_levels(
                    eigvals,
                    degeneracy=self.ensemble.eigval_degeneracy,
                    cdf=cdf,
                    dimension=self.ensemble.dimension,
                )

            for cdf, outputs in zip(
                cdf_factory.interpolators_from_coeffs(coeffs),
                self.outputs.var_unfolded_by_degree,
                strict=True,
            ):
                outputs.add_unfolded_levels(
                    eigvals,
                    degeneracy=self.ensemble.eigval_degeneracy,
                    cdf=cdf,
                    dimension=self.ensemble.dimension,
                )
