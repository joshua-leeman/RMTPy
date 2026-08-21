from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import attrs
import numpy as np

import rmtpy.conversion
from rmtpy.compounds import Compound

from ..base import Simulation
from ..observable import Observable
from ..statistics import REALIZATIONS_METADATA, truncated_polynomial_degrees
from ..unfolding import TruncatedPolynomialCdfFactory
from .outputs import TimeDelayOutputs, create_time_delay_outputs

ENERGIES_METADATA: dict[str, str] = {
    "latex_name": "E",
}


def format_energy_path_value(energy: float) -> str:
    return f"{energy:.5g}".replace("-", "n").replace(".", "p")


def normalize_energies(energies: Any) -> np.ndarray:
    energies_array = np.array(energies, dtype=np.float64, copy=True, order="C")
    if energies_array.ndim == 0:
        energies_array = energies_array.reshape(1)
    if energies_array.ndim != 1:
        raise ValueError("`energies` must be a one-dimensional array-like.")
    if energies_array.size < 1:
        raise ValueError("`energies` must contain at least one value.")
    if not np.all(np.isfinite(energies_array)):
        raise ValueError("`energies` must contain finite values.")

    energies_array[energies_array == 0.0] = 0.0
    if np.unique(energies_array).size != energies_array.size:
        raise ValueError("`energies` must contain unique values.")

    path_values = tuple(format_energy_path_value(value) for value in energies_array)
    if len(set(path_values)) != len(path_values):
        raise ValueError(
            "`energies` must remain unique when formatted with five significant "
            "digits for output paths."
        )

    energies_array.flags.writeable = False
    return energies_array


def run_time_delay_statistics(
    compound: Compound,
    *,
    realizs: int,
    energies: Iterable[float] = (0.0,),
) -> None:
    TimeDelayStatisticsSimulation(
        compound=compound,
        realizs=realizs,
        energies=energies,
    ).run()


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class TimeDelayStatisticsSimulation(Simulation):
    """Monte Carlo experiment for proper delay times at fixed probe energies."""

    compound: Compound = attrs.field(
        converter=Compound.create,
    )
    realizs: int = attrs.field(
        converter=int,
        validator=attrs.validators.gt(0),
        metadata=REALIZATIONS_METADATA,
    )
    energies: np.ndarray = attrs.field(
        default=(0.0,),
        converter=normalize_energies,
        metadata=ENERGIES_METADATA,
        repr=False,
    )

    outputs: TimeDelayOutputs = attrs.field(
        default=attrs.Factory(create_time_delay_outputs, takes_self=True),
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

    def energy_path(self, energy: float) -> Path:
        return Path(f"energy_{format_energy_path_value(energy)}")

    def observable_output_path(self, observable: Observable) -> Path:
        return self.energy_path(observable.metadata["energy"])

    def realize_monte_carlo_simulation(self) -> None:
        spectral_density = self.compound.ensemble.spectral_density
        cdf_factory = None
        avg_cdf_interpolators = ()
        if self.truncated_degrees:
            cdf_factory = TruncatedPolynomialCdfFactory(
                density=spectral_density,
                degrees=self.truncated_degrees,
                density_name="spectral",
            )
            avg_cdf_interpolators = cdf_factory.average_interpolators()

        for time_delays, eigvals in self.compound.time_delays_stream(
            energies=self.energies, realizs=self.realizs
        ):
            self.outputs.add_raw(time_delays)
            self.outputs.add_weight_unfolded(
                time_delays,
                energies=self.energies,
                cdf=spectral_density.weight_cdf,
                dimension=self.compound.ensemble.dimension,
            )

            self.outputs.add_average_unfolded(
                time_delays,
                energies=self.energies,
                cdfs=avg_cdf_interpolators,
                dimension=self.compound.ensemble.dimension,
            )

            if not self.outputs.var_unfolded_by_degree:
                continue

            if cdf_factory is None:
                raise RuntimeError("Variate unfolding requires a CDF factory.")

            coeffs = spectral_density.compute_variate_coeffs(eigvals)
            self.outputs.add_variate_unfolded(
                time_delays,
                energies=self.energies,
                cdfs=cdf_factory.interpolators_from_coeffs(coeffs),
                dimension=self.compound.ensemble.dimension,
            )
