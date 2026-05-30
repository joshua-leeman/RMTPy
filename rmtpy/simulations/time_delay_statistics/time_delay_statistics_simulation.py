from __future__ import annotations

import copy
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import attrs
import numpy as np
from scipy.interpolate import PchipInterpolator

import rmtpy.density
from rmtpy.compounds import Compound
from rmtpy.conversion import RMT_CONVERTER

from ..base import Simulation
from ..histogram import Histogram
from ..observable import Observable
from ..statistics import (
    create_truncated_cdf_interpolators,
    observable_data_list,
    simulation_output_path,
    truncate_coeffs,
    truncated_polynomial_degrees,
)
from .observables import (
    create_avg_unfolded_time_delay_histograms,
    create_time_delay_histograms,
    create_var_unfolded_time_delay_histograms,
    create_weight_unfolded_time_delay_histograms,
)

REALIZATIONS_METADATA: dict[str, str] = {
    "dir_name": "realizs",
    "latex_name": "R",
}
ENERGIES_METADATA: dict[str, str] = {
    "latex_name": "E",
}

CDF = Callable[[np.ndarray], np.ndarray]


def normalize_energies(energies: Any) -> np.ndarray:
    energies_array: np.ndarray = np.asarray(energies, dtype=np.float64)
    if energies_array.ndim == 0:
        energies_array = energies_array.reshape(1)
    if energies_array.ndim != 1:
        raise ValueError("`energies` must be a one-dimensional array-like.")
    if energies_array.size < 1:
        raise ValueError("`energies` must contain at least one value.")
    if not np.all(np.isfinite(energies_array)):
        raise ValueError("`energies` must contain finite values.")

    return np.ascontiguousarray(energies_array)


def format_energy_path_value(energy: float) -> str:
    return f"{energy:.5g}".replace("-", "m").replace(".", "p")


def run_time_delay_statistics(
    compound: Compound,
    realizs: int,
    energies: Any = (0.0,),
) -> None:
    TimeDelayStatisticsSimulation(
        compound=compound,
        realizs=realizs,
        energies=energies,
    ).run()


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
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

    time_delay_histograms: list[Observable] = attrs.field(
        default=attrs.Factory(create_time_delay_histograms, takes_self=True),
        init=False,
        repr=False,
    )
    time_delay_histograms_wgt_unfolded: list[Observable] = attrs.field(
        default=attrs.Factory(
            create_weight_unfolded_time_delay_histograms,
            takes_self=True,
        ),
        init=False,
        repr=False,
    )
    time_delay_histograms_avg_unfolded: list[Observable] = attrs.field(
        default=attrs.Factory(
            create_avg_unfolded_time_delay_histograms, takes_self=True
        ),
        init=False,
        repr=False,
    )
    time_delay_histograms_var_unfolded: list[Observable] = attrs.field(
        default=attrs.Factory(
            create_var_unfolded_time_delay_histograms, takes_self=True
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

    def energy_path(self, energy: float) -> Path:
        return Path(f"energy_{format_energy_path_value(energy)}")

    def observable_output_path(self, observable: Observable) -> Path:
        return self.energy_path(observable.metadata["energy"])

    @property
    def truncated_degrees(self) -> tuple[int, ...]:
        return tuple(
            truncated_polynomial_degrees(
                self.compound.ensemble.max_spectral_polynomial_degree
            )
        )

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

    def create_truncated_average_cdf_interpolators(self) -> list[PchipInterpolator]:
        return create_truncated_cdf_interpolators(
            self.compound.resonance_density,
            self.truncated_degrees,
            density_name="resonance",
        )

    def create_histogram_groups(
        self,
        observables: list[Observable],
    ) -> list[list[Histogram]]:
        histograms: list[Histogram] = observable_data_list(observables, Histogram)
        num_energies: int = self.energies.size
        return [
            histograms[start : start + num_energies]
            for start in range(0, len(histograms), num_energies)
        ]

    def time_delays_and_resonances_stream(
        self,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        compound: Compound = self.compound

        for _ in range(self.realizs):
            rng_state: dict[str, Any] = copy.deepcopy(compound.rng_state)
            time_delays: np.ndarray = next(
                compound.time_delays_stream(self.energies, 1)
            )
            next_rng_state: dict[str, Any] = copy.deepcopy(compound.rng_state)

            compound.set_rng_state(rng_state)
            try:
                resonances: np.ndarray = next(compound.resonance_real_parts_stream(1))
            finally:
                compound.set_rng_state(next_rng_state)

            yield time_delays, resonances

    def add_raw_contributions(
        self,
        time_delays: np.ndarray,
        histograms: list[Histogram],
    ) -> None:
        for delay_values, histogram in zip(time_delays, histograms, strict=True):
            histogram.add_histogram_contribution(delay_values)

    def compute_unfolded_time_delays(
        self,
        delay_values: np.ndarray,
        energy: float,
        cdf: CDF,
    ) -> np.ndarray:
        valid_delays: np.ndarray = delay_values[
            np.isfinite(delay_values) & (delay_values > 0.0)
        ]
        if valid_delays.size == 0:
            return valid_delays

        widths: np.ndarray = np.reciprocal(valid_delays)
        unfolded_widths: np.ndarray = rmtpy.density.unfold_widths_with_cdf(
            widths,
            np.full_like(widths, energy),
            cdf,
            self.compound.ensemble.dimension,
        )
        valid_unfolded_widths: np.ndarray = unfolded_widths[
            np.isfinite(unfolded_widths) & (unfolded_widths > 0.0)
        ]
        return np.reciprocal(valid_unfolded_widths)

    def add_unfolded_contributions(
        self,
        time_delays: np.ndarray,
        cdf: CDF,
        histograms: list[Histogram],
    ) -> None:
        for energy, delay_values, histogram in zip(
            self.energies,
            time_delays,
            histograms,
            strict=True,
        ):
            histogram.add_histogram_contribution(
                self.compute_unfolded_time_delays(delay_values, energy, cdf)
            )

    def realize_monte_carlo_simulation(self) -> None:
        resonance_density: rmtpy.density.DensityModel = self.compound.resonance_density
        truncated_degrees: tuple[int, ...] = self.truncated_degrees
        avg_cdf_interpolators: list[PchipInterpolator] | None = None

        raw_histograms: list[Histogram] = observable_data_list(
            self.time_delay_histograms,
            Histogram,
        )
        weight_histograms: list[Histogram] = observable_data_list(
            self.time_delay_histograms_wgt_unfolded,
            Histogram,
        )
        average_histogram_groups: list[list[Histogram]] = self.create_histogram_groups(
            self.time_delay_histograms_avg_unfolded
        )
        variate_histogram_groups: list[list[Histogram]] = self.create_histogram_groups(
            self.time_delay_histograms_var_unfolded
        )

        for time_delays, resonances in self.time_delays_and_resonances_stream():
            self.add_raw_contributions(time_delays, raw_histograms)
            self.add_unfolded_contributions(
                time_delays,
                resonance_density.weight_cdf,
                weight_histograms,
            )

            if average_histogram_groups and avg_cdf_interpolators is None:
                avg_cdf_interpolators = (
                    self.create_truncated_average_cdf_interpolators()
                )

            for avg_cdf_interpolator, histograms in zip(
                avg_cdf_interpolators or [],
                average_histogram_groups,
                strict=True,
            ):
                self.add_unfolded_contributions(
                    time_delays,
                    avg_cdf_interpolator,
                    histograms,
                )

            if not variate_histogram_groups:
                continue

            resonance_coeffs: np.ndarray = resonance_density.compute_variate_coeffs(
                resonances
            )
            for degree, histograms in zip(
                truncated_degrees,
                variate_histogram_groups,
                strict=True,
            ):
                var_cdf_interpolator: PchipInterpolator = (
                    resonance_density.create_variate_cdf_interpolator(
                        coeffs=truncate_coeffs(resonance_coeffs, degree)
                    )
                )
                self.add_unfolded_contributions(
                    time_delays,
                    var_cdf_interpolator,
                    histograms,
                )
