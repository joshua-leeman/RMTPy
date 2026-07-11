from __future__ import annotations

from typing import TYPE_CHECKING

import attrs
import numpy as np

from ..histogram import Histogram
from ..observable import Observable
from ..unfolding import CDF, unfold_time_delays
from .observables import (
    create_avg_unfolded_time_delay_histograms,
    create_time_delay_histograms,
    create_var_unfolded_time_delay_histograms,
    create_weight_unfolded_time_delay_histograms,
)

if TYPE_CHECKING:
    from .time_delay_statistics_simulation import TimeDelayStatisticsSimulation


def create_time_delay_outputs(
    simulation: TimeDelayStatisticsSimulation,
) -> TimeDelayOutputs:
    num_energies = simulation.energies.size
    avg_unfolded = tuple(create_avg_unfolded_time_delay_histograms(simulation))
    var_unfolded = tuple(create_var_unfolded_time_delay_histograms(simulation))

    return TimeDelayOutputs(
        raw=tuple(create_time_delay_histograms(simulation)),
        weight_unfolded=tuple(create_weight_unfolded_time_delay_histograms(simulation)),
        avg_unfolded_by_degree=group_observables_by_degree(
            avg_unfolded,
            num_energies=num_energies,
        ),
        var_unfolded_by_degree=group_observables_by_degree(
            var_unfolded,
            num_energies=num_energies,
        ),
    )


def group_observables_by_degree(
    observables: tuple[Observable[Histogram], ...],
    *,
    num_energies: int,
) -> tuple[tuple[Observable[Histogram], ...], ...]:
    if len(observables) % num_energies != 0:
        raise ValueError("Time-delay observables do not divide evenly by energy.")

    return tuple(
        observables[start : start + num_energies]
        for start in range(0, len(observables), num_energies)
    )


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class TimeDelayOutputs:
    raw: tuple[Observable[Histogram], ...]
    weight_unfolded: tuple[Observable[Histogram], ...]
    avg_unfolded_by_degree: tuple[tuple[Observable[Histogram], ...], ...]
    var_unfolded_by_degree: tuple[tuple[Observable[Histogram], ...], ...]

    def add_raw(self, time_delays: np.ndarray) -> None:
        for delay_values, observable in zip(time_delays, self.raw, strict=True):
            observable.data.add_histogram_contribution(delay_values)

    def add_weight_unfolded(
        self,
        time_delays: np.ndarray,
        *,
        energies: np.ndarray,
        cdf: CDF,
        dimension: int,
    ) -> None:
        self._add_unfolded_to(
            self.weight_unfolded,
            time_delays,
            energies=energies,
            cdf=cdf,
            dimension=dimension,
        )

    def add_average_unfolded(
        self,
        time_delays: np.ndarray,
        *,
        energies: np.ndarray,
        cdfs: tuple[CDF, ...],
        dimension: int,
    ) -> None:
        self._add_unfolded_by_degree(
            self.avg_unfolded_by_degree,
            time_delays,
            energies=energies,
            cdfs=cdfs,
            dimension=dimension,
        )

    def add_variate_unfolded(
        self,
        time_delays: np.ndarray,
        *,
        energies: np.ndarray,
        cdfs: tuple[CDF, ...],
        dimension: int,
    ) -> None:
        self._add_unfolded_by_degree(
            self.var_unfolded_by_degree,
            time_delays,
            energies=energies,
            cdfs=cdfs,
            dimension=dimension,
        )

    def _add_unfolded_to(
        self,
        observables: tuple[Observable[Histogram], ...],
        time_delays: np.ndarray,
        *,
        energies: np.ndarray,
        cdf: CDF,
        dimension: int,
    ) -> None:
        for energy, delay_values, observable in zip(
            energies,
            time_delays,
            observables,
            strict=True,
        ):
            observable.data.add_histogram_contribution(
                unfold_time_delays(
                    delay_values,
                    energy=float(energy),
                    cdf=cdf,
                    dimension=dimension,
                )
            )

    def _add_unfolded_by_degree(
        self,
        groups: tuple[tuple[Observable[Histogram], ...], ...],
        time_delays: np.ndarray,
        *,
        energies: np.ndarray,
        cdfs: tuple[CDF, ...],
        dimension: int,
    ) -> None:
        for cdf, observables in zip(cdfs, groups, strict=True):
            self._add_unfolded_to(
                observables,
                time_delays,
                energies=energies,
                cdf=cdf,
                dimension=dimension,
            )
