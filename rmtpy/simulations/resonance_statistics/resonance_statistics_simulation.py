from __future__ import annotations

from pathlib import Path

import attrs
from scipy.interpolate import PchipInterpolator

import rmtpy.conversion
from rmtpy.compounds import Compound

from ..base import Simulation
from ..statistics import REALIZATIONS_METADATA, truncated_polynomial_degrees
from ..unfolding import TruncatedPolynomialCdfFactory
from .outputs import ResonanceStatisticsOutputs, create_resonance_statistics_outputs


def run_resonance_statistics(compound: Compound, realizs: int) -> None:
    ResonanceStatisticsSimulation(compound=compound, realizs=realizs).run()


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class ResonanceStatisticsSimulation(Simulation):
    """Monte Carlo experiment for complex resonances of an open system."""

    compound: Compound = attrs.field(
        converter=Compound.create,
    )
    realizs: int = attrs.field(
        converter=int,
        validator=attrs.validators.gt(0),
        metadata=REALIZATIONS_METADATA,
    )

    outputs: ResonanceStatisticsOutputs = attrs.field(
        default=attrs.Factory(create_resonance_statistics_outputs, takes_self=True),
        init=False,
        repr=False,
    )

    @property
    def to_path(self) -> Path:
        return rmtpy.conversion.to_path(
            self,
            root=Path(self.path_name) / self.compound.to_path,
        )

    @property
    def truncated_degrees(self) -> tuple[int, ...]:
        return tuple(
            truncated_polynomial_degrees(
                self.compound.ensemble.max_spectral_polynomial_degree
            )
        )

    def create_cdf_factory(self) -> TruncatedPolynomialCdfFactory:
        return TruncatedPolynomialCdfFactory(
            density=self.compound.resonance_density,
            degrees=self.truncated_degrees,
            density_name="resonance",
        )

    def realize_monte_carlo_simulation(self) -> None:
        resonance_density = self.compound.resonance_density
        cdf_factory = self.create_cdf_factory() if self.truncated_degrees else None
        avg_cdf_interpolators: tuple[PchipInterpolator, ...] | None = None
        ensemble = self.compound.ensemble

        for complex_energies in self.compound.resonances_stream(realizs=self.realizs):
            resonances = complex_energies.real
            widths = -2 * complex_energies.imag

            self.outputs.raw.add_raw(
                resonances,
                widths,
                ensemble=ensemble,
            )
            coeffs = self.outputs.coefficients.add(
                resonances,
                density=resonance_density,
            )
            self.outputs.weight_unfolded.add_unfolded(
                resonances,
                widths,
                cdf=resonance_density.weight_cdf,
                ensemble=ensemble,
            )

            if cdf_factory is None:
                continue

            if avg_cdf_interpolators is None:
                avg_cdf_interpolators = cdf_factory.average_interpolators()

            for cdf, outputs in zip(
                avg_cdf_interpolators,
                self.outputs.avg_unfolded_by_degree,
                strict=True,
            ):
                outputs.add_unfolded(
                    resonances,
                    widths,
                    cdf=cdf,
                    ensemble=ensemble,
                )

            for cdf, outputs in zip(
                cdf_factory.interpolators_from_coeffs(coeffs),
                self.outputs.var_unfolded_by_degree,
                strict=True,
            ):
                outputs.add_unfolded(
                    resonances,
                    widths,
                    cdf=cdf,
                    ensemble=ensemble,
                )
