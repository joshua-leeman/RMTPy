from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import attrs
import numpy as np
from scipy.special import xlogy

from ..observable import Observable
from .cdo_dynamics import CDODynamicsData
from .evolved_states import EvolvedStatesData
from .observables import (
    create_cdo_dynamics_observable,
    create_evolved_states_observable,
)

if TYPE_CHECKING:
    from .cdo_evolution_simulation import CDOEvolutionSimulation

MAX_WORKSPACE_ARRAY_BYTES: int = 32 * 1024**2


def create_lower_triangle_indices(
    dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    return np.tril_indices(dimension)


def create_cdo_evolution_outputs(
    simulation: CDOEvolutionSimulation,
) -> CDOEvolutionOutputs:
    dynamics = create_cdo_dynamics_observable(
        dimension=simulation.ensemble.dimension,
        scale=simulation.time_scale,
        logD_time_support=simulation.logD_time_support,
        num_times=simulation.num_times,
    )
    evolved_states = None
    if simulation.retain_evolved_states:
        evolved_states = create_evolved_states_observable(
            realizs=simulation.realizs,
            num_times=simulation.num_times,
            dimension=simulation.ensemble.dimension,
            dtype=simulation.ensemble.complex_dtype,
        )
    return CDOEvolutionOutputs(
        dynamics=dynamics,
        evolved_states=evolved_states,
        target_realizs=simulation.realizs,
        time_chunk_size=simulation.time_chunk_size,
        complex_dtype=simulation.ensemble.complex_dtype,
    )


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class CDOEvolutionOutputs:
    """The CDO observable and its memory-adaptive Monte Carlo accumulator."""

    dynamics: Observable[CDODynamicsData]
    evolved_states: Observable[EvolvedStatesData] | None = attrs.field(repr=False)
    target_realizs: int = attrs.field(
        converter=int,
        validator=attrs.validators.gt(0),
        repr=False,
    )
    time_chunk_size: int = attrs.field(
        converter=int,
        validator=attrs.validators.gt(0),
        repr=False,
    )
    complex_dtype: np.dtype = attrs.field(converter=np.dtype, repr=False)

    _realizs_count: np.ndarray = attrs.field(init=False, repr=False)
    _log_probability_sum: np.ndarray = attrs.field(
        init=False,
        repr=False,
    )
    _lower_indices: tuple[np.ndarray, np.ndarray] = attrs.field(
        init=False,
        repr=False,
    )
    _evolved_states: np.ndarray | None = attrs.field(init=False, repr=False)
    _density_operator_lower_sum: np.ndarray | None = attrs.field(
        init=False,
        repr=False,
    )

    @_realizs_count.default
    def _create_realizs_count(self) -> np.ndarray:
        return np.zeros(1, dtype=np.int64)

    @_log_probability_sum.default
    def _create_log_probability_sum(self) -> np.ndarray:
        return np.zeros(
            (self.dynamics.data.num_times, self.dynamics.data.dimension),
            dtype=np.float64,
        )

    @_lower_indices.default
    def _create_lower_indices(self) -> tuple[np.ndarray, np.ndarray]:
        return create_lower_triangle_indices(self.dynamics.data.dimension)

    @_evolved_states.default
    def _create_evolved_states(self) -> np.ndarray | None:
        if self.evolved_states is not None:
            return self.evolved_states.data.states

        dimension = self.dynamics.data.dimension
        packed_dimension = dimension * (dimension + 1) // 2
        if self.target_realizs * dimension <= packed_dimension:
            return np.empty(
                (
                    self.target_realizs,
                    self.dynamics.data.num_times,
                    dimension,
                ),
                dtype=self.complex_dtype,
            )
        return None

    @_density_operator_lower_sum.default
    def _create_density_operator_lower_sum(self) -> np.ndarray | None:
        if self._evolved_states is not None:
            return None
        packed_dimension = len(self._lower_indices[0])
        return np.zeros(
            (self.dynamics.data.num_times, packed_dimension),
            dtype=self.complex_dtype,
        )

    @property
    def realizs(self) -> int:
        return int(self._realizs_count[0])

    @property
    def accumulation_mode(self) -> str:
        if self._evolved_states is not None:
            return "states"
        return "density_operator"

    @property
    def statistics_chunk_size(self) -> int:
        dimension = self.dynamics.data.dimension
        if self._evolved_states is not None:
            entries_per_time = max(
                self.target_realizs * dimension,
                self.target_realizs**2,
            )
        else:
            entries_per_time = dimension**2
        bytes_per_time = entries_per_time * self.complex_dtype.itemsize
        workspace_chunk_size = max(
            1,
            MAX_WORKSPACE_ARRAY_BYTES // bytes_per_time,
        )
        return min(self.time_chunk_size, workspace_chunk_size)

    def iter_observables(self) -> Iterator[Observable]:
        yield self.dynamics
        if self.evolved_states is not None:
            yield self.evolved_states

    def reset(self) -> None:
        self._realizs_count.fill(0)
        self._log_probability_sum.fill(0.0)
        if self._density_operator_lower_sum is not None:
            self._density_operator_lower_sum.fill(0.0)

        data = self.dynamics.data
        data.probabilities.fill(0.0)
        data.classical_purity.fill(0.0)
        data.quantum_purity.fill(0.0)
        data.entropy.fill(0.0)
        data.kl_divergence.fill(0.0)
        data._realizs_count.fill(0)

    def add_evolved_states(self, states: np.ndarray) -> None:
        states = np.asarray(states)
        data = self.dynamics.data
        expected_shape = (data.num_times, data.dimension)
        if states.shape != expected_shape:
            raise ValueError(
                f"Evolved states must have shape {expected_shape}, got {states.shape}."
            )
        if not np.all(np.isfinite(states)):
            raise ValueError("Evolved states must contain finite values.")
        if self.realizs >= self.target_realizs:
            raise RuntimeError(
                "The CDO accumulator already contains the requested number of "
                "realizations."
            )

        lower_rows, lower_columns = self._lower_indices
        chunk_size = self.statistics_chunk_size
        for start in range(0, data.num_times, chunk_size):
            stop = min(start + chunk_size, data.num_times)
            state_chunk = states[start:stop]
            probabilities = np.abs(state_chunk) ** 2
            normalizations = np.sum(probabilities, axis=1, keepdims=True)
            if np.any(~np.isfinite(normalizations)) or np.any(normalizations <= 0.0):
                raise ValueError("Every evolved state must have a positive finite norm.")
            probabilities /= normalizations
            log_probabilities = np.full_like(probabilities, -np.inf)
            np.log(
                probabilities,
                out=log_probabilities,
                where=probabilities > 0.0,
            )
            self._log_probability_sum[start:stop] += log_probabilities

            if self._density_operator_lower_sum is not None:
                self._density_operator_lower_sum[start:stop] += (
                    state_chunk[:, lower_rows]
                    * state_chunk[:, lower_columns].conj()
                    / normalizations
                )

        if self._evolved_states is not None:
            self._evolved_states[self.realizs] = states
        self._realizs_count[0] += 1

    def calculate_dynamics(self) -> None:
        data = self.dynamics.data
        if self.realizs == 0:
            data.probabilities.fill(0.0)
            data.classical_purity.fill(0.0)
            data.quantum_purity.fill(0.0)
            data.entropy.fill(0.0)
            data.kl_divergence.fill(0.0)
            data._realizs_count.fill(0)
            return

        chunk_size = self.statistics_chunk_size
        for start in range(0, data.num_times, chunk_size):
            stop = min(start + chunk_size, data.num_times)

            probabilities, eigenvalues, eigenproblem_dimension = (
                self._probabilities_and_eigenvalues_chunk(start, stop)
            )
            np.clip(probabilities, 0.0, None, out=probabilities)
            probabilities /= np.sum(probabilities, axis=1, keepdims=True)
            data.probabilities[start:stop] = probabilities
            data.classical_purity[start:stop] = np.sum(probabilities**2, axis=1)

            tolerance = (
                np.finfo(eigenvalues.dtype).eps
                * eigenproblem_dimension
                * np.max(np.abs(eigenvalues), axis=1)
            )
            if np.any(eigenvalues < -tolerance[:, np.newaxis]):
                raise FloatingPointError(
                    "A chaotic density operator has a significantly negative eigenvalue."
                )

            eigenvalues[eigenvalues < tolerance[:, np.newaxis]] = 0.0
            eigenvalues /= np.sum(eigenvalues, axis=1, keepdims=True)
            data.quantum_purity[start:stop] = np.sum(eigenvalues**2, axis=1)
            data.entropy[start:stop] = -np.sum(
                xlogy(eigenvalues, eigenvalues),
                axis=1,
            )

            mean_log_probability = self._log_probability_sum[start:stop] / self.realizs
            weighted_mean_log_probability = np.zeros_like(probabilities)
            np.multiply(
                probabilities,
                mean_log_probability,
                out=weighted_mean_log_probability,
                where=probabilities > 0.0,
            )
            data.kl_divergence[start:stop] = np.sum(
                xlogy(probabilities, probabilities) - weighted_mean_log_probability,
                axis=1,
            )

        roundoff_tolerance = (
            32.0 * np.finfo(data.kl_divergence.dtype).eps * data.dimension
        )
        small_negative = (data.kl_divergence < 0.0) & (
            data.kl_divergence >= -roundoff_tolerance
        )
        data.kl_divergence[small_negative] = 0.0
        if np.any(data.kl_divergence < 0.0):
            raise FloatingPointError(
                "The ensemble-averaged Kullback-Leibler divergence became negative."
            )
        data._realizs_count[0] = self.realizs

    def _probabilities_and_eigenvalues_chunk(
        self,
        start: int,
        stop: int,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        data = self.dynamics.data
        if self._evolved_states is not None:
            states = self._evolved_states[: self.realizs, start:stop]
            probabilities = np.abs(states) ** 2
            normalizations = np.sum(probabilities, axis=2)
            normalized_states = states / np.sqrt(normalizations)[..., np.newaxis]
            probabilities = np.mean(np.abs(normalized_states) ** 2, axis=0)
            gram_matrices = (
                np.einsum(
                    "rti,sti->trs",
                    normalized_states.conj(),
                    normalized_states,
                    optimize=True,
                )
                / self.realizs
            )
            return (
                probabilities,
                np.linalg.eigvalsh(gram_matrices),
                self.realizs,
            )

        if self._density_operator_lower_sum is None:
            raise RuntimeError("The CDO accumulator has no density representation.")

        lower_rows, lower_columns = self._lower_indices
        lower_triangle = self._density_operator_lower_sum[start:stop] / self.realizs
        density_operators = np.zeros(
            (stop - start, data.dimension, data.dimension),
            dtype=self.complex_dtype,
        )
        density_operators[:, lower_rows, lower_columns] = lower_triangle
        density_operators[:, lower_columns, lower_rows] = lower_triangle.conj()
        diagonal = np.arange(data.dimension)
        density_operators[:, diagonal, diagonal] = density_operators[
            :, diagonal, diagonal
        ].real
        traces = np.trace(density_operators, axis1=1, axis2=2).real
        if np.any(~np.isfinite(traces)) or np.any(traces <= 0.0):
            raise FloatingPointError(
                "Chaotic density operators must have positive finite traces."
            )
        density_operators /= traces[:, np.newaxis, np.newaxis]
        probabilities = np.diagonal(
            density_operators,
            axis1=1,
            axis2=2,
        ).real.copy()
        return (
            probabilities,
            np.linalg.eigvalsh(density_operators),
            data.dimension,
        )
