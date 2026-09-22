# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportUnusedCallResult=false, reportPrivateUsage=false, reportImplicitStringConcatenation=false, reportMissingParameterType=false, reportUnnecessaryIsInstance=false, reportImplicitOverride=false, reportExplicitAny=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
from scipy.special import xlogy

from rmtpy.ensembles import GOE, GUE
from rmtpy.simulations.base_simulation import SimulationExecutionState as ExecutionState
from rmtpy.simulations.cdo_evolution import (
    CDOEvolutionRequest,
    CDOEvolutionResult,
    CDOEvolutionSimulation,
    load_cdo_evolution_result,
    plot_cdo_evolution_result,
    run_cdo_evolution,
    save_cdo_evolution_result,
)
from rmtpy.simulations.cdo_evolution.accumulator import (
    CDONumericalAccumulator,
    classical_kl,
    clip_kl_divergence_roundoff,
    compute_classical_kl_divergences,
)
from rmtpy.simulations.cdo_evolution.cdo_dynamics import (
    CDOInformationPlot,
    CDOKLRatioPlot,
    CDOProbabilitiesPlot,
    CDOPuritiesPlot,
    build_cdo_time_grid,
)
from rmtpy.simulations.cdo_evolution.cdo_evolution_simulation import (
    mixed_estimate_probabilities,
)
from rmtpy.simulations.cdo_evolution.data_factories import (
    build_kl_divergence_histogram_data,
)
from rmtpy.simulations.cdo_evolution.kl_divergence_histogram import (
    KLDivergenceHistogramData,
    KLDivergenceHistogramPlot,
)
from rmtpy.simulations.histogram import Histogram
from rmtpy.simulations.statistics import LOG_D_TIME_SUPPORT


def require[T](value: T | None) -> T:
    assert value is not None
    return value


def direct_analysis(states: np.ndarray) -> tuple[np.ndarray, ...]:
    states = np.asarray(states)
    normalized_states = states / np.sqrt(
        np.sum(np.abs(states) ** 2, axis=2, keepdims=True)
    )
    density_operators = np.einsum(
        "rti,rtj->tij",
        normalized_states,
        normalized_states.conj(),
        optimize=True,
    ) / len(states)
    density_operators /= np.real(
        np.trace(
            density_operators,
            axis1=1,
            axis2=2,
        )
    )[:, np.newaxis, np.newaxis]

    mixed_probabilities = np.real(
        np.diagonal(
            density_operators,
            axis1=1,
            axis2=2,
        )
    ).copy()
    mixed_probabilities /= np.sum(mixed_probabilities, axis=1, keepdims=True)
    classical_purity = np.sum(mixed_probabilities**2, axis=1)

    eigenvalues = np.linalg.eigvalsh(density_operators)
    eigenvalues[eigenvalues < 0.0] = 0.0
    eigenvalues /= np.sum(eigenvalues, axis=1, keepdims=True)
    quantum_purity = np.sum(eigenvalues**2, axis=1)
    entropy = -np.sum(xlogy(eigenvalues, eigenvalues), axis=1)
    return (
        mixed_probabilities,
        classical_purity,
        quantum_purity,
        entropy,
    )


def direct_evolution(
    *,
    ensemble,
    realizs: int,
    times: np.ndarray,
    heisenberg_time: float,
    initial_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    evolved_states = []
    heisenberg_states = []
    for eigenvalues, eigenvectors in ensemble.eigsys_stream(realizs):
        rotated_state = eigenvectors.conj().T @ initial_state
        evolved_states.append(
            (np.exp(-1j * np.outer(times, eigenvalues)) * rotated_state) @ eigenvectors.T
        )
        heisenberg_states.append(
            (np.exp(-1j * heisenberg_time * eigenvalues) * rotated_state) @ eigenvectors.T
        )
    return np.asarray(evolved_states), np.asarray(heisenberg_states)


def normalized_probabilities(states: np.ndarray) -> np.ndarray:
    probabilities = np.abs(states) ** 2
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
    return probabilities


def uniform_estimate(dimension: int, num_times: int | None = None) -> np.ndarray:
    value = np.full(dimension, 1.0 / dimension, dtype=np.float64)
    if num_times is None:
        return value
    return np.repeat(value[np.newaxis, :], num_times, axis=0)


def direct_mean_kl(estimate: np.ndarray, states: np.ndarray) -> np.ndarray:
    return np.mean(classical_kl(estimate, normalized_probabilities(states)), axis=0)


def direct_heisenberg_kl(
    estimate: np.ndarray,
    states: np.ndarray,
) -> np.ndarray:
    return classical_kl(estimate, normalized_probabilities(states))


def mixed_estimates(
    *,
    num_majoranas: int,
    goe_seed: int,
    gue_seed: int,
    initial_state: np.ndarray,
    times: np.ndarray | None,
    heisenberg_time: float | None,
    realizs: int,
    time_chunk_size: int,
    dtype: str | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    goe_kwargs: dict[str, Any] = {"num_majoranas": num_majoranas, "seed": goe_seed}
    gue_kwargs: dict[str, Any] = {"num_majoranas": num_majoranas, "seed": gue_seed}
    if dtype is not None:
        goe_kwargs["dtype"] = dtype
        gue_kwargs["dtype"] = dtype
    goe_grid, goe_heisenberg = mixed_estimate_probabilities(
        GOE(**goe_kwargs),
        initial_state,
        times=times,
        heisenberg_time=heisenberg_time,
        realizs=realizs,
        time_chunk_size=time_chunk_size,
    )
    gue_grid, gue_heisenberg = mixed_estimate_probabilities(
        GUE(**gue_kwargs),
        initial_state,
        times=times,
        heisenberg_time=heisenberg_time,
        realizs=realizs,
        time_chunk_size=time_chunk_size,
    )
    return goe_grid, gue_grid, goe_heisenberg, gue_heisenberg


def histogram_counts(samples: np.ndarray, bins: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(bins, samples, side="right") - 1
    valid = (indices >= 0) & (indices < len(bins) - 1)
    counts = np.zeros(len(bins) - 1, dtype=np.int64)
    np.add.at(counts, indices[valid], 1)
    return counts


class CDOAccumulatorTests(unittest.TestCase):
    @staticmethod
    def fixture_states() -> tuple[np.ndarray, np.ndarray]:
        states = np.array(
            [
                [
                    [1.0 + 0.0j, 0.5 + 0.25j],
                    [0.8 + 0.1j, 0.4 - 0.3j],
                    [0.7 - 0.2j, 0.6 + 0.1j],
                    [0.5 + 0.4j, 0.9 - 0.2j],
                    [0.3 - 0.1j, 1.1 + 0.2j],
                ],
                [
                    [0.4 + 0.2j, 1.0 - 0.1j],
                    [0.6 - 0.2j, 0.7 + 0.4j],
                    [0.9 + 0.3j, 0.5 - 0.2j],
                    [1.0 - 0.2j, 0.4 + 0.1j],
                    [0.8 + 0.1j, 0.6 - 0.4j],
                ],
            ],
            dtype=np.complex128,
        )
        heisenberg_states = np.array(
            [[0.9 + 0.2j, 0.4 - 0.1j], [0.3 - 0.2j, 1.1 + 0.1j]],
            dtype=np.complex128,
        )
        return states, heisenberg_states

    def build_accumulator(
        self,
        *,
        retain_evolved_states: bool,
        goe_estimate: np.ndarray | None = None,
        gue_estimate: np.ndarray | None = None,
        goe_heisenberg: np.ndarray | None = None,
        gue_heisenberg: np.ndarray | None = None,
    ) -> CDONumericalAccumulator:
        if goe_estimate is None:
            goe_estimate = uniform_estimate(2, 5)
        if gue_estimate is None:
            gue_estimate = uniform_estimate(2, 5)
        if goe_heisenberg is None:
            goe_heisenberg = uniform_estimate(2)
        if gue_heisenberg is None:
            gue_heisenberg = uniform_estimate(2)
        return CDONumericalAccumulator(
            dimension=2,
            num_times=5,
            target_realizs=2,
            complex_dtype=np.complex128,
            time_chunk_size=2,
            include_dynamics=True,
            include_heisenberg_kl=True,
            retain_evolved_states=retain_evolved_states,
            goe_estimate_probabilities=goe_estimate,
            gue_estimate_probabilities=gue_estimate,
            goe_heisenberg_estimate=goe_heisenberg,
            gue_heisenberg_estimate=gue_heisenberg,
        )

    def assert_analysis_matches(
        self,
        accumulator: CDONumericalAccumulator,
        states: np.ndarray,
        heisenberg_states: np.ndarray,
        goe_estimate: np.ndarray,
        gue_estimate: np.ndarray,
        goe_heisenberg: np.ndarray,
        gue_heisenberg: np.ndarray,
    ) -> None:
        result = accumulator.finalize()
        self.assertIsNotNone(result.dynamics)
        if result.dynamics is None:
            self.fail("Expected time-dependent CDO analysis.")
        expected = direct_analysis(states)
        for actual, reference in zip(
            (
                result.dynamics.probabilities,
                result.dynamics.classical_purity,
                result.dynamics.quantum_purity,
                result.dynamics.entropy,
            ),
            expected,
            strict=True,
        ):
            self.assertEqual(actual.dtype, np.dtype(np.float64))
            np.testing.assert_allclose(actual, reference, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(
            result.dynamics.kl_goe,
            direct_mean_kl(goe_estimate, states),
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            result.dynamics.kl_gue,
            direct_mean_kl(gue_estimate, states),
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            require(result.heisenberg_kl_goe),
            direct_heisenberg_kl(goe_heisenberg, heisenberg_states),
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            require(result.heisenberg_kl_gue),
            direct_heisenberg_kl(gue_heisenberg, heisenberg_states),
            rtol=1e-13,
            atol=1e-13,
        )

    def test_explicit_calculation_agrees_in_both_internal_modes(self) -> None:
        states, heisenberg_states = self.fixture_states()
        goe_estimate = uniform_estimate(2, 5)
        gue_estimate = np.array([[0.8, 0.2]] * 5, dtype=np.float64)
        goe_heisenberg = uniform_estimate(2)
        gue_heisenberg = np.array([0.8, 0.2], dtype=np.float64)
        density_accumulator = self.build_accumulator(
            retain_evolved_states=False,
            goe_estimate=goe_estimate,
            gue_estimate=gue_estimate,
            goe_heisenberg=goe_heisenberg,
            gue_heisenberg=gue_heisenberg,
        )
        state_accumulator = self.build_accumulator(
            retain_evolved_states=True,
            goe_estimate=goe_estimate,
            gue_estimate=gue_estimate,
            goe_heisenberg=goe_heisenberg,
            gue_heisenberg=gue_heisenberg,
        )

        density_accumulator.add(states[0], heisenberg_state=heisenberg_states[0])
        self.assertEqual(density_accumulator.accumulation_mode, "density_operator")
        self.assertEqual(
            density_accumulator._density_operator_lower_sum.shape,
            (5, 3),
        )
        lower_rows, lower_columns = np.tril_indices(2)
        normalizations = np.sum(np.abs(states[0]) ** 2, axis=1, keepdims=True)
        expected_lower_triangle = (
            states[0][:, lower_rows] * states[0][:, lower_columns].conj() / normalizations
        )
        np.testing.assert_allclose(
            require(density_accumulator._density_operator_lower_sum),
            expected_lower_triangle,
        )

        state_accumulator.add(states[0], heisenberg_state=heisenberg_states[0])
        for accumulator in (density_accumulator, state_accumulator):
            accumulator.add(states[1], heisenberg_state=heisenberg_states[1])
            self.assertEqual(accumulator.statistics_chunk_size, 2)
            self.assert_analysis_matches(
                accumulator,
                states,
                heisenberg_states,
                goe_estimate,
                gue_estimate,
                goe_heisenberg,
                gue_heisenberg,
            )

        self.assertEqual(state_accumulator.accumulation_mode, "states")
        state_result = state_accumulator.finalize()
        self.assertIsNotNone(state_result.evolved_states)
        np.testing.assert_array_equal(state_result.evolved_states, states)
        self.assertEqual(state_result.evolved_states.dtype, np.dtype(np.complex128))

        density_result = density_accumulator.finalize()
        self.assertIsNone(density_result.evolved_states)
        for state_values, density_values in zip(
            (
                state_result.dynamics.probabilities,
                state_result.dynamics.classical_purity,
                state_result.dynamics.quantum_purity,
                state_result.dynamics.entropy,
                state_result.dynamics.kl_goe,
                state_result.dynamics.kl_gue,
            ),
            (
                density_result.dynamics.probabilities,
                density_result.dynamics.classical_purity,
                density_result.dynamics.quantum_purity,
                density_result.dynamics.entropy,
                density_result.dynamics.kl_goe,
                density_result.dynamics.kl_gue,
            ),
            strict=True,
        ):
            np.testing.assert_allclose(state_values, density_values, atol=1e-13)

    def test_mode_boundary_selective_storage_and_workspace_plan(self) -> None:
        boundary = CDONumericalAccumulator(
            dimension=3,
            num_times=4,
            target_realizs=2,
            complex_dtype=np.complex128,
            time_chunk_size=4,
            include_heisenberg_kl=False,
        )
        self.assertEqual(boundary.target_realizs * boundary.dimension, 6)
        self.assertEqual(boundary.packed_dimension, 6)
        self.assertEqual(boundary.accumulation_mode, "states")

        density = CDONumericalAccumulator(
            dimension=3,
            num_times=4,
            target_realizs=3,
            complex_dtype=np.complex128,
            time_chunk_size=4,
            include_heisenberg_kl=False,
            max_workspace_array_bytes=np.dtype(np.complex128).itemsize * 3**2,
        )
        self.assertEqual(density.accumulation_mode, "density_operator")
        self.assertEqual(density.statistics_chunk_size, 1)
        self.assertEqual(density._density_operator_lower_sum.shape, (4, 6))
        self.assertIsNone(density._state_buffer)

        state_rd_dominated = CDONumericalAccumulator(
            dimension=8,
            num_times=5,
            target_realizs=2,
            complex_dtype=np.complex128,
            time_chunk_size=5,
            include_heisenberg_kl=False,
            max_workspace_array_bytes=2 * 2 * 8 * np.dtype(np.complex128).itemsize,
        )
        self.assertEqual(state_rd_dominated.accumulation_mode, "states")
        self.assertEqual(state_rd_dominated.statistics_chunk_size, 2)
        self.assertIsNone(state_rd_dominated._lower_indices)

        state_r2_dominated = CDONumericalAccumulator(
            dimension=2,
            num_times=5,
            target_realizs=5,
            complex_dtype=np.complex128,
            time_chunk_size=5,
            include_heisenberg_kl=False,
            retain_evolved_states=True,
            max_workspace_array_bytes=5**2 * np.dtype(np.complex128).itemsize - 1,
        )
        self.assertEqual(state_r2_dominated.statistics_chunk_size, 1)
        self.assertIsNone(state_r2_dominated._lower_indices)

        histogram_only = CDONumericalAccumulator(
            dimension=3,
            num_times=4,
            target_realizs=3,
            complex_dtype=np.complex128,
            time_chunk_size=2,
            include_dynamics=False,
            include_heisenberg_kl=True,
            goe_heisenberg_estimate=uniform_estimate(3),
            gue_heisenberg_estimate=uniform_estimate(3),
        )
        self.assertIsNone(histogram_only.accumulation_mode)
        self.assertIsNone(histogram_only._kl_goe_sum)
        self.assertIsNone(histogram_only._state_buffer)
        self.assertIsNone(histogram_only._density_operator_lower_sum)
        self.assertIsNone(histogram_only._lower_indices)
        self.assertEqual(require(histogram_only._heisenberg_kl_goe).shape, (3,))
        self.assertEqual(require(histogram_only._heisenberg_kl_gue).shape, (3,))

        goe_heisenberg = np.array([0.7, 0.3], dtype=np.float64)
        gue_heisenberg = np.array([0.4, 0.6], dtype=np.float64)
        per_realization = CDONumericalAccumulator(
            dimension=2,
            num_times=1,
            target_realizs=5,
            complex_dtype=np.complex128,
            time_chunk_size=5,
            include_dynamics=False,
            goe_heisenberg_estimate=goe_heisenberg,
            gue_heisenberg_estimate=gue_heisenberg,
        )
        heisenberg_states = np.array(
            [[value, 1.0] for value in range(1, 6)],
            dtype=np.complex128,
        )
        for state in heisenberg_states:
            per_realization.add(None, heisenberg_state=state)
        per_realization_result = per_realization.finalize()
        np.testing.assert_allclose(
            require(per_realization_result.heisenberg_kl_goe),
            direct_heisenberg_kl(goe_heisenberg, heisenberg_states),
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            require(per_realization_result.heisenberg_kl_gue),
            direct_heisenberg_kl(gue_heisenberg, heisenberg_states),
            rtol=1e-13,
            atol=1e-13,
        )

        states_only = CDONumericalAccumulator(
            dimension=3,
            num_times=4,
            target_realizs=3,
            complex_dtype=np.complex64,
            time_chunk_size=2,
            include_dynamics=False,
            include_heisenberg_kl=False,
            retain_evolved_states=True,
        )
        self.assertEqual(states_only.accumulation_mode, "states")
        self.assertIsNone(states_only._kl_goe_sum)
        self.assertIsNone(states_only._heisenberg_kl_goe)
        self.assertIsNone(states_only._lower_indices)
        self.assertEqual(states_only._state_buffer.dtype, np.dtype(np.complex64))

    def test_time_zero_rank_one_and_missing_support(self) -> None:
        matching_estimate = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float64)
        matching_heisenberg = np.array([1.0, 0.0], dtype=np.float64)
        rank_one = CDONumericalAccumulator(
            dimension=2,
            num_times=2,
            target_realizs=1,
            complex_dtype=np.complex128,
            time_chunk_size=1,
            goe_estimate_probabilities=matching_estimate,
            gue_estimate_probabilities=matching_estimate,
            goe_heisenberg_estimate=matching_heisenberg,
            gue_heisenberg_estimate=matching_heisenberg,
        )
        rank_one.add(
            np.array([[1.0, 0.0], [1.0, 1.0j]]),
            heisenberg_state=np.array([1.0, 0.0]),
        )
        result = rank_one.finalize()
        np.testing.assert_array_equal(result.dynamics.probabilities[0], [1.0, 0.0])
        np.testing.assert_allclose(result.dynamics.classical_purity, [1.0, 0.5])
        np.testing.assert_allclose(result.dynamics.quantum_purity, 1.0)
        np.testing.assert_allclose(result.dynamics.entropy, 0.0, atol=1e-15)
        np.testing.assert_allclose(result.dynamics.kl_goe, 0.0, atol=1e-15)
        np.testing.assert_allclose(result.dynamics.kl_gue, 0.0, atol=1e-15)
        np.testing.assert_array_equal(result.heisenberg_kl_goe, [0.0])
        np.testing.assert_array_equal(result.heisenberg_kl_gue, [0.0])

        uniform = uniform_estimate(2, 1)
        uniform_heisenberg = uniform_estimate(2)
        missing_support = CDONumericalAccumulator(
            dimension=2,
            num_times=1,
            target_realizs=2,
            complex_dtype=np.complex128,
            time_chunk_size=1,
            goe_estimate_probabilities=uniform,
            gue_estimate_probabilities=uniform,
            goe_heisenberg_estimate=uniform_heisenberg,
            gue_heisenberg_estimate=uniform_heisenberg,
        )
        missing_support.add(
            np.array([[1.0, 0.0]]),
            heisenberg_state=np.array([1.0, 0.0]),
        )
        missing_support.add(
            np.array([[0.0, 1.0]]),
            heisenberg_state=np.array([0.0, 1.0]),
        )
        missing_result = missing_support.finalize()
        np.testing.assert_allclose(missing_result.dynamics.probabilities[0], [0.5, 0.5])
        self.assertTrue(np.isposinf(missing_result.dynamics.kl_goe[0]))
        self.assertTrue(np.isposinf(missing_result.dynamics.kl_gue[0]))
        self.assertTrue(np.all(np.isposinf(require(missing_result.heisenberg_kl_goe))))
        self.assertTrue(np.all(np.isposinf(require(missing_result.heisenberg_kl_gue))))

    def test_validation_is_atomic_and_capacity_is_enforced(self) -> None:
        def create() -> CDONumericalAccumulator:
            return CDONumericalAccumulator(
                dimension=2,
                num_times=2,
                target_realizs=1,
                complex_dtype=np.complex128,
                time_chunk_size=1,
                goe_estimate_probabilities=uniform_estimate(2, 2),
                gue_estimate_probabilities=uniform_estimate(2, 2),
                goe_heisenberg_estimate=uniform_estimate(2),
                gue_heisenberg_estimate=uniform_estimate(2),
            )

        invalid_cases = (
            (np.ones((1, 2)), np.ones(2)),
            (np.array([[1.0, 0.0], [np.nan, 1.0]]), np.ones(2)),
            (np.array([[1.0, 0.0], [0.0, 0.0]]), np.ones(2)),
            (np.ones((2, 2)), np.ones((1, 2))),
            (np.ones((2, 2)), np.array([1.0, np.inf])),
            (np.ones((2, 2)), np.zeros(2)),
        )
        for states, heisenberg_state in invalid_cases:
            with self.subTest(states=states, heisenberg_state=heisenberg_state):
                accumulator = create()
                with self.assertRaises(ValueError):
                    accumulator.add(states, heisenberg_state=heisenberg_state)
                self.assertEqual(accumulator.realizs, 0)
                np.testing.assert_array_equal(
                    require(accumulator._kl_goe_sum),
                    np.zeros(2),
                )
                np.testing.assert_array_equal(
                    require(accumulator._heisenberg_kl_goe),
                    np.zeros(1),
                )

        accumulator = create()
        states = np.ones((2, 2))
        heisenberg_state = np.ones(2)
        accumulator.add(states, heisenberg_state=heisenberg_state)
        with self.assertRaises(RuntimeError):
            accumulator.add(states, heisenberg_state=heisenberg_state)

        histogram_only = CDONumericalAccumulator(
            dimension=2,
            num_times=1,
            target_realizs=1,
            complex_dtype=np.complex128,
            time_chunk_size=1,
            include_dynamics=False,
            goe_heisenberg_estimate=uniform_estimate(2),
            gue_heisenberg_estimate=uniform_estimate(2),
        )
        with self.assertRaises(ValueError):
            histogram_only.add(np.ones((1, 2)), heisenberg_state=np.ones(2))

        dynamics_only = CDONumericalAccumulator(
            dimension=2,
            num_times=1,
            target_realizs=1,
            complex_dtype=np.complex128,
            time_chunk_size=1,
            include_heisenberg_kl=False,
        )
        with self.assertRaises(ValueError):
            dynamics_only.add(np.ones((1, 2)), heisenberg_state=np.ones(2))

    def test_finalize_is_idempotent_then_add_and_reset_are_clean(self) -> None:
        accumulator = CDONumericalAccumulator(
            dimension=2,
            num_times=1,
            target_realizs=2,
            complex_dtype=np.complex128,
            time_chunk_size=1,
            retain_evolved_states=True,
            goe_estimate_probabilities=uniform_estimate(2, 1),
            gue_estimate_probabilities=uniform_estimate(2, 1),
            goe_heisenberg_estimate=uniform_estimate(2),
            gue_heisenberg_estimate=uniform_estimate(2),
        )
        first_state = np.array([[1.0, 1.0j]])
        second_state = np.array([[1.0, -1.0j]])
        accumulator.add(first_state, heisenberg_state=first_state[0])
        partial = accumulator.finalize()
        self.assertIs(accumulator.finalize(), partial)
        self.assertEqual(partial.realizs, 1)
        partial_states = partial.evolved_states.copy()

        accumulator.add(second_state, heisenberg_state=second_state[0])
        complete = accumulator.finalize()
        self.assertIsNot(complete, partial)
        self.assertIs(accumulator.finalize(), complete)
        self.assertEqual(complete.realizs, 2)
        np.testing.assert_array_equal(partial.evolved_states, partial_states)

        state_buffer = accumulator._state_buffer
        kl_goe_sum = accumulator._kl_goe_sum
        heisenberg_kl_goe = accumulator._heisenberg_kl_goe
        accumulator.reset()
        self.assertEqual(accumulator.realizs, 0)
        self.assertIs(accumulator._state_buffer, state_buffer)
        self.assertIs(accumulator._kl_goe_sum, kl_goe_sum)
        self.assertIs(accumulator._heisenberg_kl_goe, heisenberg_kl_goe)
        empty = accumulator.finalize()
        np.testing.assert_array_equal(empty.dynamics.probabilities, np.zeros((1, 2)))
        self.assertEqual(empty.heisenberg_kl_goe.shape, (0,))
        self.assertEqual(empty.heisenberg_kl_gue.shape, (0,))
        self.assertEqual(empty.evolved_states.shape, (0, 1, 2))
        accumulator.add(second_state, heisenberg_state=second_state[0])
        refilled = accumulator.finalize()
        self.assertEqual(refilled.realizs, 1)
        np.testing.assert_array_equal(refilled.evolved_states[0], second_state)

    def test_rounding_negatives_are_clipped_and_material_negatives_raise(self) -> None:
        eps64 = np.finfo(np.float64).eps
        small = np.array([-16.0 * eps64 * 2, 0.25], dtype=np.float64)
        clip_kl_divergence_roundoff(
            small,
            dimension=2,
            description="test KL divergence",
        )
        self.assertEqual(small[0], 0.0)

        material = np.array([-64.0 * eps64 * 2], dtype=np.float64)
        with self.assertRaises(FloatingPointError):
            clip_kl_divergence_roundoff(
                material,
                dimension=2,
                description="test KL divergence",
            )

        float32_scale = np.array([-np.finfo(np.float32).eps / 4], dtype=np.float64)
        clip_kl_divergence_roundoff(
            float32_scale,
            dimension=2,
            description="single-precision KL divergence",
            roundoff_dtype=np.dtype(np.float32),
        )
        self.assertEqual(float32_scale[0], 0.0)

        accumulator = CDONumericalAccumulator(
            dimension=2,
            num_times=1,
            target_realizs=1,
            complex_dtype=np.complex128,
            time_chunk_size=1,
            include_heisenberg_kl=False,
        )
        accumulator.add(np.ones((1, 2)), heisenberg_state=None)
        probabilities = np.array([[0.5, 0.5]])
        rounding_spectrum = np.array([[-eps64, 1.0 + eps64]])
        with patch.object(
            CDONumericalAccumulator,
            "_probabilities_and_eigenvalues_chunk",
            return_value=(probabilities.copy(), rounding_spectrum.copy(), 2),
        ):
            np.testing.assert_allclose(
                accumulator.finalize().dynamics.quantum_purity, 1.0
            )

        invalid_accumulator = CDONumericalAccumulator(
            dimension=2,
            num_times=1,
            target_realizs=1,
            complex_dtype=np.complex128,
            time_chunk_size=1,
            include_heisenberg_kl=False,
        )
        invalid_accumulator.add(np.ones((1, 2)), heisenberg_state=None)
        invalid_spectrum = np.array([[-1e-6, 1.000001]])
        with (
            patch.object(
                CDONumericalAccumulator,
                "_probabilities_and_eigenvalues_chunk",
                return_value=(probabilities.copy(), invalid_spectrum, 2),
            ),
            self.assertRaises(FloatingPointError),
        ):
            invalid_accumulator.finalize()

    def test_single_precision_outputs_use_published_dtypes(self) -> None:
        integer_input = CDONumericalAccumulator(
            dimension=2,
            num_times=1,
            target_realizs=1,
            complex_dtype=np.complex64,
            time_chunk_size=1,
            retain_evolved_states=True,
            goe_estimate_probabilities=uniform_estimate(2, 1),
            gue_estimate_probabilities=uniform_estimate(2, 1),
            goe_heisenberg_estimate=uniform_estimate(2),
            gue_heisenberg_estimate=uniform_estimate(2),
        )
        integer_input.add(
            np.array([[1, 0]], dtype=np.int64),
            heisenberg_state=np.array([1, 0], dtype=np.int64),
        )
        integer_result = integer_input.finalize()
        self.assertEqual(integer_result.evolved_states.dtype, np.dtype(np.complex64))
        np.testing.assert_array_equal(integer_result.dynamics.probabilities, [[1.0, 0.0]])

        simulation = CDOEvolutionSimulation(
            ensemble=GOE(num_majoranas=4, dtype="float32", seed=1),
            goe_ensemble=GOE(num_majoranas=4, dtype="float32", seed=2),
            gue_ensemble=GUE(num_majoranas=4, dtype="float32", seed=3),
            realizs=1,
            num_times=2,
            request={"quantities": ("dynamics", "evolved_states")},
        )
        result = simulation.execute()
        self.assertEqual(result.dynamics.probabilities.dtype, np.dtype(np.float64))
        self.assertEqual(result.dynamics.kl_goe.dtype, np.dtype(np.float64))
        self.assertEqual(result.dynamics.kl_gue.dtype, np.dtype(np.float64))
        self.assertEqual(result.evolved_states.states.dtype, np.dtype(np.complex64))


class CDOEvolutionTests(unittest.TestCase):
    def assert_histogram_matches_samples(
        self,
        *,
        bins: np.ndarray,
        counts: np.ndarray,
        histogram: np.ndarray,
        samples: np.ndarray,
        realizs: int,
    ) -> None:
        expected_counts = histogram_counts(samples, bins)
        self.assertEqual(realizs, len(samples))
        np.testing.assert_array_equal(counts, expected_counts)
        if np.sum(expected_counts) == 0:
            np.testing.assert_array_equal(histogram, np.zeros(len(counts)))
            return
        expected_density = expected_counts / (
            np.sum(expected_counts) * np.diff(bins)
        )
        np.testing.assert_allclose(histogram, expected_density)
        self.assertAlmostEqual(
            float(np.sum(histogram * np.diff(bins))),
            1.0,
        )

    def assert_kl_histogram_matches(
        self,
        data: KLDivergenceHistogramData,
        goe_samples: np.ndarray,
        gue_samples: np.ndarray,
    ) -> None:
        self.assert_histogram_matches_samples(
            bins=data.bins,
            counts=data.goe_counts,
            histogram=data.goe_histogram,
            samples=goe_samples,
            realizs=data.realizs,
        )
        self.assert_histogram_matches_samples(
            bins=data.bins,
            counts=data.gue_counts,
            histogram=data.gue_histogram,
            samples=gue_samples,
            realizs=data.realizs,
        )

    def test_request_is_canonical_and_rejects_legacy_selection(self) -> None:
        default = CDOEvolutionRequest()
        self.assertEqual(
            default.quantities,
            ("dynamics", "kl_divergence_histogram"),
        )
        request = CDOEvolutionRequest(quantities=("evolved_states", "dynamics"))
        self.assertEqual(request.quantities, ("dynamics", "evolved_states"))
        self.assertEqual(
            request.json_normalized(),
            {"quantities": ["dynamics", "evolved_states"]},
        )
        self.assertIs(CDOEvolutionRequest.create(request), request)
        self.assertEqual(
            CDOEvolutionRequest.create({"quantities": "dynamics"}).quantities,
            ("dynamics",),
        )

        for invalid in (
            (),
            ("dynamics", "dynamics"),
            ("retain_evolved_states",),
            ("unknown",),
            (1,),
        ):
            with (
                self.subTest(quantities=invalid),
                self.assertRaises((TypeError, ValueError)),
            ):
                CDOEvolutionRequest(quantities=cast(Any, invalid))
        with self.assertRaises(TypeError):
            CDOEvolutionRequest.create(cast(Any, 1))
        with self.assertRaises(TypeError):
            CDOEvolutionSimulation(
                ensemble=GOE(num_majoranas=4),
                realizs=1,
                **cast(Any, {"retain_evolved_states": True}),
            )

    def test_initial_state_and_time_grid_are_validated_and_immutable(self) -> None:
        source = np.array([0.0, 1.0])
        simulation = CDOEvolutionSimulation(
            ensemble=GOE(num_majoranas=4),
            realizs=1,
            initial_state=source,
            num_times=4,
        )
        source[1] = 0.0
        np.testing.assert_array_equal(simulation.initial_state, [0.0, 1.0])
        self.assertFalse(np.shares_memory(source, simulation.initial_state))
        self.assertFalse(simulation.initial_state.flags.writeable)
        with self.assertRaises(ValueError):
            simulation.initial_state[0] = 1.0

        times = build_cdo_time_grid(
            dimension=simulation.ensemble.dimension,
            scale=simulation.time_scale,
            logD_time_support=simulation.logD_time_support,
            num_times=simulation.num_times,
        )
        self.assertEqual(times.dtype, np.dtype(np.float64))
        self.assertEqual(times.shape, (4,))
        self.assertEqual(times[0], 0.0)
        self.assertTrue(np.all(times[1:] > 0.0))
        self.assertFalse(times.flags.writeable)

        explicit_times = build_cdo_time_grid(
            dimension=4,
            scale=2.0,
            logD_time_support=(-1.0, 1.0),
            num_times=4,
        )
        np.testing.assert_array_equal(explicit_times, [0.0, 0.5, 2.0, 8.0])

        for invalid_state in (
            [1.0],
            [1.0, np.nan],
            [1.0, 1.0],
            [0.0, 0.0],
        ):
            with (
                self.subTest(initial_state=invalid_state),
                self.assertRaises(ValueError),
            ):
                CDOEvolutionSimulation(
                    ensemble=GOE(num_majoranas=4),
                    realizs=1,
                    initial_state=cast(Any, invalid_state),
                )

    def test_default_execute_returns_typed_result_and_is_one_shot(self) -> None:
        simulation = CDOEvolutionSimulation(
            ensemble=GOE(num_majoranas=4, seed=123),
            realizs=2,
            num_times=4,
            time_chunk_size=2,
        )
        with self.assertRaises(RuntimeError):
            _ = simulation.result
        initial_rng_state = simulation.ensemble.rng_state.copy()

        result = simulation.execute()
        self.assertIsInstance(result, CDOEvolutionResult)
        self.assertIs(simulation.result, result)
        self.assertEqual(simulation.execution_state, ExecutionState.COMPLETE)
        self.assertFalse(hasattr(simulation, "outputs"))
        self.assertEqual(
            result.context.execution["accumulation_mode"], "density_operator"
        )
        self.assertEqual(result.context.execution["estimate_realizs"], 2)
        self.assertIn("goe_rng", result.context.execution)
        self.assertIn("gue_rng", result.context.execution)
        self.assertEqual(result.context.rng["initial_state"], initial_rng_state)
        self.assertEqual(
            result.context.rng["final_state"],
            simulation.ensemble.rng_state,
        )
        self.assertEqual(
            result.context.output_request,
            {"quantities": ["dynamics", "kl_divergence_histogram"]},
        )
        self.assertEqual(
            tuple(data.file_name for data in result.iterate_data()),
            ("cdo_dynamics_data", "kl_divergence_histogram_data"),
        )
        self.assertIsNotNone(result.dynamics)
        self.assertIsNotNone(result.kl_divergence_histogram)
        self.assertIsNone(result.evolved_states)
        with self.assertRaises(RuntimeError):
            simulation.execute()

    def test_failed_execution_is_terminal(self) -> None:
        simulation = CDOEvolutionSimulation(
            ensemble=GOE(num_majoranas=4, seed=4),
            realizs=1,
            num_times=2,
        )
        with (
            patch.object(
                type(simulation.ensemble),
                "eigsys_stream",
                side_effect=RuntimeError("eigensystem failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "eigensystem failure"),
        ):
            simulation.execute()
        self.assertEqual(simulation.execution_state, ExecutionState.FAILED)
        with self.assertRaises(RuntimeError):
            _ = simulation.result
        with self.assertRaises(RuntimeError):
            simulation.execute()

    def test_seeded_dynamics_and_exact_heisenberg_histogram_in_both_modes(
        self,
    ) -> None:
        cases = (
            ("density_operator", 4, 2, 11, 101, 202),
            ("states", 6, 2, 10, 303, 404),
        )
        for (
            expected_mode,
            num_majoranas,
            realizs,
            truth_seed,
            goe_seed,
            gue_seed,
        ) in cases:
            with self.subTest(accumulation_mode=expected_mode):
                simulation = CDOEvolutionSimulation(
                    ensemble=GUE(num_majoranas=num_majoranas, seed=truth_seed),
                    goe_ensemble=GOE(num_majoranas=num_majoranas, seed=goe_seed),
                    gue_ensemble=GUE(num_majoranas=num_majoranas, seed=gue_seed),
                    realizs=realizs,
                    estimate_realizs=realizs,
                    num_times=4,
                    logD_time_support=(-1.0, 0.0),
                    time_chunk_size=2,
                )
                times = build_cdo_time_grid(
                    dimension=simulation.ensemble.dimension,
                    scale=simulation.time_scale,
                    logD_time_support=simulation.logD_time_support,
                    num_times=simulation.num_times,
                )
                self.assertFalse(np.any(np.isclose(times, simulation.heisenberg_time)))
                reference_ensemble = GUE(num_majoranas=num_majoranas, seed=truth_seed)
                expected_states, expected_heisenberg_states = direct_evolution(
                    ensemble=reference_ensemble,
                    realizs=realizs,
                    times=times,
                    heisenberg_time=simulation.heisenberg_time,
                    initial_state=simulation.initial_state,
                )
                expected_dynamics = direct_analysis(expected_states)
                goe_grid, gue_grid, goe_heisenberg, gue_heisenberg = mixed_estimates(
                    num_majoranas=num_majoranas,
                    goe_seed=goe_seed,
                    gue_seed=gue_seed,
                    initial_state=simulation.initial_state,
                    times=times,
                    heisenberg_time=simulation.heisenberg_time,
                    realizs=realizs,
                    time_chunk_size=2,
                )
                expected_kl_goe = direct_mean_kl(require(goe_grid), expected_states)
                expected_kl_gue = direct_mean_kl(require(gue_grid), expected_states)
                expected_heisenberg_goe = direct_heisenberg_kl(
                    require(goe_heisenberg),
                    expected_heisenberg_states,
                )
                expected_heisenberg_gue = direct_heisenberg_kl(
                    require(gue_heisenberg),
                    expected_heisenberg_states,
                )

                with patch(
                    "rmtpy.simulations.cdo_evolution.cdo_evolution_simulation."
                    "build_kl_divergence_histogram_data",
                    wraps=build_kl_divergence_histogram_data,
                ) as histogram_factory:
                    result = simulation.execute()
                self.assertEqual(
                    result.context.execution["accumulation_mode"], expected_mode
                )
                np.testing.assert_array_equal(result.dynamics.times, times)
                for actual, reference in zip(
                    (
                        result.dynamics.probabilities,
                        result.dynamics.classical_purity,
                        result.dynamics.quantum_purity,
                        result.dynamics.entropy,
                    ),
                    expected_dynamics,
                    strict=True,
                ):
                    np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(
                    result.dynamics.kl_goe,
                    expected_kl_goe,
                    rtol=1e-12,
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    result.dynamics.kl_gue,
                    expected_kl_gue,
                    rtol=1e-12,
                    atol=1e-12,
                )
                self.assertEqual(result.dynamics.realizs, realizs)
                self.assertEqual(
                    result.context.rng["final_state"],
                    reference_ensemble.rng_state,
                )
                self.assert_kl_histogram_matches(
                    result.kl_divergence_histogram,
                    expected_heisenberg_goe,
                    expected_heisenberg_gue,
                )
                np.testing.assert_allclose(
                    histogram_factory.call_args.kwargs["goe_divergences"],
                    expected_heisenberg_goe,
                    rtol=1e-12,
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    histogram_factory.call_args.kwargs["gue_divergences"],
                    expected_heisenberg_gue,
                    rtol=1e-12,
                    atol=1e-12,
                )
                self.assertEqual(
                    result.kl_divergence_histogram.metadata["heisenberg_time"],
                    simulation.heisenberg_time,
                )

                _, nearest_states = direct_evolution(
                    ensemble=GUE(num_majoranas=num_majoranas, seed=truth_seed),
                    realizs=realizs,
                    times=times,
                    heisenberg_time=times[-1],
                    initial_state=simulation.initial_state,
                )
                nearest_probabilities = normalized_probabilities(nearest_states)
                exact_probabilities = normalized_probabilities(
                    expected_heisenberg_states
                )
                self.assertFalse(np.allclose(nearest_probabilities, exact_probabilities))
                nearest_goe = direct_heisenberg_kl(require(goe_grid)[-1], nearest_states)
                self.assertFalse(
                    np.array_equal(
                        result.kl_divergence_histogram.goe_counts,
                        histogram_counts(
                            nearest_goe,
                            result.kl_divergence_histogram.bins,
                        ),
                    )
                )

    def test_reverse_kl_orientation_is_preserved(self) -> None:
        estimate = np.array([0.7, 0.3])
        realization_probabilities = np.array([[0.9, 0.1], [0.2, 0.8]])
        estimate_first = compute_classical_kl_divergences(
            estimate,
            realization_probabilities,
        )
        realization_first = compute_classical_kl_divergences(
            realization_probabilities,
            estimate,
        )
        np.testing.assert_allclose(
            estimate_first,
            np.sum(
                estimate * np.log(estimate / realization_probabilities),
                axis=1,
            ),
        )
        self.assertFalse(np.allclose(estimate_first, realization_first))
        self.assertTrue(np.isfinite(classical_kl([0.0, 1.0], [0.5, 0.5])))
        self.assertTrue(np.isposinf(classical_kl([0.5, 0.5], [1.0, 0.0])))
        skipped_zero = classical_kl([0.0, 1.0], [0.0, 1.0])
        self.assertEqual(skipped_zero, 0.0)

    def test_selective_results_keep_only_required_products(self) -> None:
        configurations = (
            (("dynamics",), (True, False, False), "density_operator"),
            (("kl_divergence_histogram",), (False, True, False), None),
            (("evolved_states",), (False, False, True), "states"),
            (
                ("dynamics", "kl_divergence_histogram", "evolved_states"),
                (True, True, True),
                "states",
            ),
        )
        results = {}
        for quantities, presence, expected_mode in configurations:
            with self.subTest(quantities=quantities):
                simulation = CDOEvolutionSimulation(
                    ensemble=GUE(num_majoranas=4, seed=29),
                    goe_ensemble=GOE(num_majoranas=4, seed=101),
                    gue_ensemble=GUE(num_majoranas=4, seed=202),
                    realizs=2,
                    num_times=5,
                    time_chunk_size=2,
                    request={"quantities": quantities},
                )
                result = simulation.execute()
                results[quantities] = result
                self.assertEqual(
                    result.context.execution["accumulation_mode"], expected_mode
                )
                self.assertEqual(
                    (
                        result.dynamics is not None,
                        result.kl_divergence_histogram is not None,
                        result.evolved_states is not None,
                    ),
                    presence,
                )

        dynamics_only = results[("dynamics",)]
        all_outputs = results[("dynamics", "kl_divergence_histogram", "evolved_states")]
        for actual, reference in zip(
            (
                all_outputs.dynamics.probabilities,
                all_outputs.dynamics.classical_purity,
                all_outputs.dynamics.quantum_purity,
                all_outputs.dynamics.entropy,
                all_outputs.dynamics.kl_goe,
                all_outputs.dynamics.kl_gue,
            ),
            (
                dynamics_only.dynamics.probabilities,
                dynamics_only.dynamics.classical_purity,
                dynamics_only.dynamics.quantum_purity,
                dynamics_only.dynamics.entropy,
                dynamics_only.dynamics.kl_goe,
                dynamics_only.dynamics.kl_gue,
            ),
            strict=True,
        ):
            np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(
            all_outputs.kl_divergence_histogram.goe_counts,
            results[("kl_divergence_histogram",)].kl_divergence_histogram.goe_counts,
        )
        np.testing.assert_array_equal(
            all_outputs.kl_divergence_histogram.gue_counts,
            results[("kl_divergence_histogram",)].kl_divergence_histogram.gue_counts,
        )
        np.testing.assert_allclose(
            all_outputs.evolved_states.states,
            results[("evolved_states",)].evolved_states.states,
        )
        np.testing.assert_allclose(
            np.sum(np.abs(all_outputs.evolved_states.states) ** 2, axis=2),
            1.0,
        )

    def test_unrequested_result_factories_are_not_called(self) -> None:
        with (
            patch(
                "rmtpy.simulations.cdo_evolution.cdo_evolution_simulation."
                "build_cdo_dynamics_data",
                side_effect=AssertionError("dynamics factory called"),
            ),
            patch(
                "rmtpy.simulations.cdo_evolution.cdo_evolution_simulation."
                "build_cdo_time_grid",
                side_effect=AssertionError("unused time grid created"),
            ),
        ):
            result = CDOEvolutionSimulation(
                ensemble=GOE(num_majoranas=4, seed=9),
                realizs=1,
                num_times=2,
                request={"quantities": ("kl_divergence_histogram",)},
            ).execute()
        self.assertIsNone(result.dynamics)

        with patch(
            "rmtpy.simulations.cdo_evolution.cdo_evolution_simulation."
            "build_kl_divergence_histogram_data",
            side_effect=AssertionError("histogram factory called"),
        ):
            result = CDOEvolutionSimulation(
                ensemble=GOE(num_majoranas=4, seed=9),
                realizs=1,
                num_times=2,
                request={"quantities": ("dynamics",)},
            ).execute()
        self.assertIsNone(result.kl_divergence_histogram)

    def test_histogram_support_is_half_open(self) -> None:
        samples = np.array([0.0, np.nextafter(5.0, 0.0), 5.0, np.inf])
        with patch.object(
            Histogram,
            "add_histogram_contribution",
            side_effect=AssertionError("scalar histogram loop used"),
        ):
            histogram = build_kl_divergence_histogram_data(
                goe_divergences=samples,
                gue_divergences=samples,
                heisenberg_time=7.0,
            )
        self.assertEqual(histogram.support, (0.0, 5.0))
        self.assertEqual(histogram.realizs, 4)
        self.assertEqual(int(np.sum(histogram.goe_counts)), 2)
        self.assertEqual(histogram.goe_counts[0], 1)
        self.assertEqual(histogram.goe_counts[-1], 1)
        self.assertEqual(int(np.sum(histogram.gue_counts)), 2)
        self.assertEqual(histogram.gue_counts[0], 1)
        self.assertEqual(histogram.gue_counts[-1], 1)
        self.assertEqual(histogram.metadata["heisenberg_time"], 7.0)
        self.assertAlmostEqual(
            float(np.sum(histogram.goe_histogram * np.diff(histogram.bins))),
            1.0,
        )

    def test_persistence_round_trip_includes_retained_states(self) -> None:
        simulation = CDOEvolutionSimulation(
            ensemble=GOE(num_majoranas=4, seed=31),
            goe_ensemble=GOE(num_majoranas=4, seed=41),
            gue_ensemble=GUE(num_majoranas=4, seed=51),
            realizs=2,
            num_times=4,
            request={
                "quantities": (
                    "dynamics",
                    "kl_divergence_histogram",
                    "evolved_states",
                )
            },
        )
        result = simulation.execute()
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = save_cdo_evolution_result(result, out_dir=tmp_dir)
            restored = load_cdo_evolution_result(run_dir)

        restored_dynamics = restored.dynamics
        restored_histogram = restored.kl_divergence_histogram
        restored_states = restored.evolved_states
        self.assertEqual(restored.context, result.context)
        for name in (
            "times",
            "probabilities",
            "classical_purity",
            "quantum_purity",
            "entropy",
            "kl_goe",
            "kl_gue",
        ):
            np.testing.assert_array_equal(
                getattr(restored_dynamics, name),
                getattr(result.dynamics, name),
            )
        self.assertEqual(restored_dynamics.realizs, simulation.realizs)
        self.assertEqual(restored_dynamics.dimension, result.dynamics.dimension)
        self.assertEqual(restored_dynamics.scale, result.dynamics.scale)
        self.assertEqual(
            restored_dynamics.logD_time_support,
            result.dynamics.logD_time_support,
        )
        self.assertEqual(restored_dynamics.num_times, result.dynamics.num_times)
        np.testing.assert_array_equal(
            restored_histogram.bins,
            result.kl_divergence_histogram.bins,
        )
        np.testing.assert_array_equal(
            restored_histogram.goe_counts,
            result.kl_divergence_histogram.goe_counts,
        )
        np.testing.assert_array_equal(
            restored_histogram.gue_counts,
            result.kl_divergence_histogram.gue_counts,
        )
        np.testing.assert_array_equal(
            restored_histogram.goe_histogram,
            result.kl_divergence_histogram.goe_histogram,
        )
        np.testing.assert_array_equal(
            restored_histogram.gue_histogram,
            result.kl_divergence_histogram.gue_histogram,
        )
        self.assertEqual(restored_histogram.realizs, simulation.realizs)
        self.assertEqual(
            restored_histogram.metadata,
            result.kl_divergence_histogram.metadata,
        )
        np.testing.assert_array_equal(
            restored_states.states, result.evolved_states.states
        )
        self.assertEqual(restored_states.states.dtype, result.evolved_states.states.dtype)
        self.assertEqual(
            restored_states.states.shape,
            (simulation.realizs, simulation.num_times, simulation.ensemble.dimension),
        )

    def test_result_consumer_routes_all_and_selective_plot_views(self) -> None:
        result = CDOEvolutionSimulation(
            ensemble=GOE(num_majoranas=4, seed=7),
            goe_ensemble=GOE(num_majoranas=4, seed=8),
            gue_ensemble=GUE(num_majoranas=4, seed=9),
            realizs=2,
            num_times=3,
        ).execute()
        with (
            patch.object(CDOProbabilitiesPlot, "plot", autospec=True) as probabilities,
            patch.object(CDOPuritiesPlot, "plot", autospec=True) as purities,
            patch.object(CDOInformationPlot, "plot", autospec=True) as information,
            patch.object(CDOKLRatioPlot, "plot", autospec=True) as ratio,
            patch.object(
                KLDivergenceHistogramPlot,
                "plot",
                autospec=True,
            ) as divergence_histogram,
        ):
            plot_cdo_evolution_result(result, out_dir=Path("unused"))

        probabilities.assert_called_once()
        purities.assert_called_once()
        information.assert_called_once()
        ratio.assert_called_once()
        divergence_histogram.assert_called_once()
        self.assertEqual(
            probabilities.call_args.kwargs["path"],
            Path("unused") / "cdo_dynamics",
        )
        self.assertEqual(
            ratio.call_args.kwargs["path"],
            Path("unused") / "cdo_dynamics",
        )
        self.assertEqual(
            divergence_histogram.call_args.kwargs["path"],
            Path("unused") / "kl_divergence_histogram",
        )

        with (
            patch.object(CDOProbabilitiesPlot, "plot", autospec=True) as probabilities,
            patch.object(CDOPuritiesPlot, "plot", autospec=True) as purities,
            patch.object(CDOInformationPlot, "plot", autospec=True) as information,
            patch.object(CDOKLRatioPlot, "plot", autospec=True) as ratio,
            patch.object(KLDivergenceHistogramPlot, "plot", autospec=True) as histogram,
        ):
            plot_cdo_evolution_result(
                result,
                out_dir=Path("unused"),
                views="cdo_purities",
            )
        probabilities.assert_not_called()
        purities.assert_called_once()
        information.assert_not_called()
        ratio.assert_not_called()
        histogram.assert_not_called()

        with (
            patch.object(CDOKLRatioPlot, "plot", autospec=True) as ratio,
            patch.object(CDOInformationPlot, "plot", autospec=True) as information,
        ):
            plot_cdo_evolution_result(
                result,
                out_dir=Path("unused"),
                views="cdo_kl_ratio",
            )
        ratio.assert_called_once()
        information.assert_not_called()

        for invalid_views in ((), ("missing",)):
            with (
                self.subTest(views=invalid_views),
                self.assertRaises((ValueError, LookupError)),
            ):
                plot_cdo_evolution_result(
                    result,
                    out_dir=Path("unused"),
                    views=invalid_views,
                )

    def test_all_plot_axis_conventions_and_plotting_preserves_live_rng(self) -> None:
        simulation = CDOEvolutionSimulation(
            ensemble=GOE(num_majoranas=4, seed=2468),
            realizs=2,
            num_times=7,
            time_chunk_size=2,
        )
        control = CDOEvolutionSimulation(
            ensemble=GOE(num_majoranas=4, seed=2468),
            realizs=2,
            num_times=7,
            time_chunk_size=2,
        )
        result = simulation.execute()
        simulation_config = deepcopy(result.context.simulation_config)
        control.execute()
        reference_axes = CDOProbabilitiesPlot(
            data=result.dynamics,
            context=result.context,
        ).axes
        expected_xlim = tuple(
            simulation.time_scale * simulation.ensemble.dimension**value
            for value in LOG_D_TIME_SUPPORT
        )

        time_plots = (
            CDOProbabilitiesPlot(
                data=result.dynamics,
                context=result.context,
            ),
            CDOPuritiesPlot(
                data=result.dynamics,
                context=result.context,
            ),
            CDOInformationPlot(
                data=result.dynamics,
                context=result.context,
            ),
        )
        dimension = simulation.ensemble.dimension
        for plot in time_plots:
            with patch.object(type(plot), "finish_plot"):
                plot.plot(path="unused")
            self.assertEqual(plot.axes.xlabel, reference_axes.xlabel)
            self.assertEqual(plot.axes.xtick_labels, reference_axes.xtick_labels)
            np.testing.assert_allclose(plot.xlim, expected_xlim)
            np.testing.assert_allclose(
                plot.axes.xticks,
                simulation.time_scale * dimension ** np.array([0.0, 0.5, 1.0]),
            )
            self.assertEqual(plot.ax.get_xscale(), "log")
            self.assertEqual(
                cast(Any, plot.ax.xaxis.get_major_locator())._base,
                dimension,
            )
            np.testing.assert_allclose(
                plot.axes.xticks,
                simulation.time_scale * dimension ** np.array([0.0, 0.5, 1.0]),
            )
            np.testing.assert_allclose(
                plot.xlim,
                simulation.time_scale * dimension ** np.array([-0.5, 1.5]),
            )
            np.testing.assert_array_equal(
                plot.ax.lines[0].get_xdata(),
                result.dynamics.times[1:],
            )
        probability_plot, purity_plot, information_plot = time_plots
        self.assertEqual(probability_plot.ax.get_yscale(), "log")
        self.assertEqual(purity_plot.ax.get_yscale(), "log")
        self.assertEqual(information_plot.ax.get_yscale(), "linear")
        self.assertEqual(
            cast(Any, probability_plot.ax.yaxis.get_major_locator())._base,
            dimension,
        )
        self.assertEqual(
            cast(Any, purity_plot.ax.yaxis.get_major_locator())._base,
            dimension,
        )
        expected_purity_ticks = dimension ** np.array([-2.0, -1.0, 0.0])
        np.testing.assert_allclose(probability_plot.axes.yticks, expected_purity_ticks)
        np.testing.assert_allclose(purity_plot.axes.yticks, expected_purity_ticks)
        np.testing.assert_allclose(
            information_plot.axes.yticks,
            np.log(dimension) * np.array([0.0, 0.5, 1.0]),
        )
        self.assertEqual(
            len(probability_plot.ax.lines), simulation.ensemble.dimension + 1
        )
        self.assertEqual(len(purity_plot.ax.lines), 2)
        self.assertEqual(len(information_plot.ax.lines), 3)
        self.assertEqual(probability_plot.file_name, "cdo_probabilities_plot")
        self.assertEqual(purity_plot.file_name, "cdo_purities_plot")
        self.assertEqual(information_plot.file_name, "cdo_information_plot")

        ratio_plot = CDOKLRatioPlot(
            data=result.dynamics,
            context=result.context,
        )
        with patch.object(type(ratio_plot), "finish_plot"):
            ratio_plot.plot(path="unused")
        self.assertEqual(ratio_plot.axes.xlabel, reference_axes.xlabel)
        self.assertEqual(ratio_plot.ax.get_xscale(), "log")
        self.assertEqual(ratio_plot.ax.get_yscale(), "linear")
        self.assertIsNone(ratio_plot.ylim)
        self.assertEqual(len(ratio_plot.ax.lines), 2)
        np.testing.assert_allclose(ratio_plot.ax.lines[1].get_ydata(), 1.0)
        self.assertEqual(ratio_plot.file_name, "cdo_kl_ratio_plot")

        histogram_plot = KLDivergenceHistogramPlot(
            data=result.kl_divergence_histogram,
            context=result.context,
        )
        with patch.object(type(histogram_plot), "finish_plot") as finish_plot:
            histogram_plot.plot(path="unused")
        finish_plot.assert_called_once_with(path="unused")
        self.assertIn(r"q(t_{\mathrm{H}})", histogram_plot.axes.xlabel)
        self.assertIn(r"p_r(t_{\mathrm{H}})", histogram_plot.axes.xlabel)
        self.assertEqual(histogram_plot.axes.ylabel, r"$P(D_{\mathrm{KL}})$")
        self.assertEqual(histogram_plot.xlim, (0.0, 5.0))
        self.assertEqual(histogram_plot.ax.get_xscale(), "linear")
        self.assertEqual(histogram_plot.ax.get_yscale(), "linear")
        self.assertEqual(
            len(histogram_plot.ax.patches),
            2 * result.kl_divergence_histogram.num_bins,
        )
        expected_edges = np.concatenate(
            (
                result.kl_divergence_histogram.bins[:-1],
                result.kl_divergence_histogram.bins[:-1],
            )
        )
        np.testing.assert_allclose(
            [cast(Any, bar).get_x() for bar in histogram_plot.ax.patches],
            expected_edges,
            atol=1e-15,
        )
        self.assertEqual(histogram_plot.legend.title, simulation.ensemble.to_latex)
        self.assertEqual(histogram_plot.file_name, "kl_divergence_histogram_plot")

        with (
            patch.object(CDOProbabilitiesPlot, "finish_plot"),
            patch.object(CDOPuritiesPlot, "finish_plot"),
            patch.object(CDOInformationPlot, "finish_plot"),
            patch.object(CDOKLRatioPlot, "finish_plot"),
            patch.object(KLDivergenceHistogramPlot, "finish_plot"),
        ):
            plot_cdo_evolution_result(result, out_dir=Path("unused"))
        self.assertEqual(
            simulation.ensemble.rng_state,
            control.ensemble.rng_state,
        )
        self.assertEqual(result.context.simulation_config, simulation_config)

    def test_execute_is_data_only_and_convenience_runner_returns_result(self) -> None:
        simulation = CDOEvolutionSimulation(
            ensemble=GOE(num_majoranas=4, seed=51),
            realizs=1,
            num_times=2,
        )
        with patch("rmtpy.simulations.persistence.save_run") as save_result:
            result = simulation.execute()
        save_result.assert_not_called()
        self.assertIs(simulation.result, result)

        with tempfile.TemporaryDirectory() as tmp_dir:
            result_without_io = CDOEvolutionSimulation(
                ensemble=GOE(num_majoranas=4, seed=52),
                realizs=1,
                num_times=2,
            ).execute()
            self.assertIsInstance(result_without_io, CDOEvolutionResult)
            self.assertEqual(tuple(Path(tmp_dir).iterdir()), ())

        sentinel = object()
        request = CDOEvolutionRequest(quantities=("dynamics",))
        with patch.object(
            CDOEvolutionSimulation,
            "execute",
            autospec=True,
            return_value=sentinel,
        ) as execute:
            returned = run_cdo_evolution(
                GOE(num_majoranas=4),
                1,
                num_times=2,
                request=request,
            )
        self.assertIs(returned, sentinel)
        called_simulation = execute.call_args.args[0]
        self.assertEqual(called_simulation.request, request)


if __name__ == "__main__":
    unittest.main()
