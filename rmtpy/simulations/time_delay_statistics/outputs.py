from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import attrs
import numpy as np

from ..histogram import Histogram
from ..observable import Observable
from ..unfolding import CDF, unfold_time_delays
from .observables import (
    create_time_delay_histograms,
    create_unfolded_time_delay_histograms,
)

if TYPE_CHECKING:
    from .time_delay_statistics_simulation import TimeDelayStatisticsSimulation


def create_time_delay_outputs(
    simulation: TimeDelayStatisticsSimulation,
) -> TimeDelayOutputs:
    return TimeDelayOutputs(
        raw=tuple(create_time_delay_histograms(simulation)),
        weight_unfolded=tuple(
            create_unfolded_time_delay_histograms(
                simulation=simulation,
                file_name_prefix="time_delay_histogram_weight_unfolded",
                unfolding="wgt",
            )
        ),
        avg_unfolded_by_degree=tuple(
            tuple(
                create_unfolded_time_delay_histograms(
                    simulation=simulation,
                    file_name_prefix="time_delay_histogram_avg_unfolded",
                    unfolding="avg",
                    degree=degree,
                )
            )
            for degree in simulation.truncated_degrees
        ),
        var_unfolded_by_degree=tuple(
            tuple(
                create_unfolded_time_delay_histograms(
                    simulation=simulation,
                    file_name_prefix="time_delay_histogram_var_unfolded",
                    unfolding="var",
                    degree=degree,
                )
            )
            for degree in simulation.truncated_degrees
        ),
    )


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class TimeDelayOutputs:
    """Time-delay histograms grouped by unfolding, degree, then energy."""

    raw: tuple[Observable[Histogram], ...]
    weight_unfolded: tuple[Observable[Histogram], ...]
    avg_unfolded_by_degree: tuple[tuple[Observable[Histogram], ...], ...]
    var_unfolded_by_degree: tuple[tuple[Observable[Histogram], ...], ...]

    def iter_observables(self) -> Iterator[Observable]:
        yield from self.raw
        yield from self.weight_unfolded
        for observables in self.avg_unfolded_by_degree:
            yield from observables
        for observables in self.var_unfolded_by_degree:
            yield from observables

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
