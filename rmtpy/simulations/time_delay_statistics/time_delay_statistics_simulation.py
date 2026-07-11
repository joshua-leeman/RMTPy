from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import attrs
import numpy as np

from rmtpy.compounds import Compound
from rmtpy.conversion import RMT_CONVERTER

from ..base import Simulation
from ..observable import Observable
from ..statistics import (
    REALIZATIONS_METADATA,
    simulation_output_path,
    truncated_polynomial_degrees,
)
from ..unfolding import TruncatedPolynomialCdfFactory
from .outputs import TimeDelayOutputs, create_time_delay_outputs

ENERGIES_METADATA: dict[str, str] = {
    "latex_name": "E",
}


def create_cdf_factory(
    sim: TimeDelayStatisticsSimulation,
) -> TruncatedPolynomialCdfFactory:
    return TruncatedPolynomialCdfFactory(
        density=sim.compound.ensemble.spectral_density,
        degrees=sim.truncated_degrees,
        density_name="spectral",
    )


def create_outputs(
    sim: TimeDelayStatisticsSimulation,
) -> TimeDelayOutputs:
    return create_time_delay_outputs(sim)


def create_truncated_degrees(
    sim: TimeDelayStatisticsSimulation,
) -> range:
    return truncated_polynomial_degrees(
        sim.compound.ensemble.max_spectral_polynomial_degree
    )


def format_energy_path_value(energy: float) -> str:
    return f"{energy:.5g}".replace("-", "n").replace(".", "p")


def normalize_energies(energies: Any) -> np.ndarray:
    energies_array = np.asarray(energies, dtype=np.float64)
    if energies_array.ndim == 0:
        energies_array = energies_array.reshape(1)
    if energies_array.ndim != 1:
        raise ValueError("`energies` must be a one-dimensional array-like.")
    if energies_array.size < 1:
        raise ValueError("`energies` must contain at least one value.")
    if not np.all(np.isfinite(energies_array)):
        raise ValueError("`energies` must contain finite values.")

    return np.ascontiguousarray(energies_array)


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

    truncated_degrees: tuple[int, ...] = attrs.field(
        default=attrs.Factory(create_truncated_degrees, takes_self=True),
        converter=tuple,
        init=False,
        repr=False,
    )
    cdf_factory: TruncatedPolynomialCdfFactory = attrs.field(
        default=attrs.Factory(create_cdf_factory, takes_self=True),
        init=False,
        repr=False,
    )
    outputs: TimeDelayOutputs = attrs.field(
        default=attrs.Factory(create_outputs, takes_self=True),
        init=False,
        repr=False,
    )

    @property
    def to_path(self) -> Path:
        return simulation_output_path(
            self,
            Path(self.path_name) / self.compound.to_path,
        )

    def energy_path(self, energy: float) -> Path:
        return Path(f"energy_{format_energy_path_value(energy)}")

    def observable_output_path(self, observable: Observable) -> Path:
        return self.energy_path(observable.metadata["energy"])

    def populate_metadata(self) -> None:
        super().populate_metadata()
        self.metadata["args"]["compound"] = RMT_CONVERTER.unstructure(self.compound)
        self.metadata["args"]["realizs"] = self.realizs
        self.metadata["args"]["energies"] = self.energies.tolist()

    def save_data(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        self.save_metadata(out_dir)

        for observable in self.iter_observables():
            observable.save_data(out_dir / self.observable_output_path(observable))

    def save_plots(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        for observable in self.iter_observables():
            observable.initialize_plot()
            observable.save_plot(out_dir / self.observable_output_path(observable))

    def realize_monte_carlo_simulation(self) -> None:
        avg_cdf_interpolators = self.cdf_factory.average_interpolators()

        for time_delays, eigvals in self.compound.time_delays_stream(
            energies=self.energies, realizs=self.realizs
        ):
            self.outputs.add_raw(time_delays)
            self.outputs.add_weight_unfolded(
                time_delays,
                energies=self.energies,
                cdf=self.compound.resonance_density.weight_cdf,
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

            coeffs = self.compound.resonance_density.compute_variate_coeffs(eigvals)
            self.outputs.add_variate_unfolded(
                time_delays,
                energies=self.energies,
                cdfs=self.cdf_factory.interpolators_from_coeffs(coeffs),
                dimension=self.compound.ensemble.dimension,
            )
