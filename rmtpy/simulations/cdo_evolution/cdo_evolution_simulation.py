from __future__ import annotations

from pathlib import Path
from typing import Any

import attrs
import numpy as np
from scipy.special import jn_zeros

import rmtpy.conversion
import rmtpy.validators
from rmtpy.ensembles import ManyBodyEnsemble

from ..base import Simulation
from ..statistics import REALIZATIONS_METADATA
from .cdo_dynamics import LOGD_TIME_SUPPORT, NUM_TIMES
from .outputs import CDOEvolutionOutputs, create_cdo_evolution_outputs

FIRST_J1_ZERO: float = float(jn_zeros(1, 1)[0])

TIME_CHUNK_SIZE: int = 64


def create_default_initial_state(simulation: CDOEvolutionSimulation) -> np.ndarray:
    state = np.zeros(
        simulation.ensemble.dimension,
        dtype=simulation.ensemble.complex_dtype,
    )
    state[0] = 1.0
    return state


def normalize_initial_state(
    initial_state: Any,
    simulation: CDOEvolutionSimulation,
) -> np.ndarray:
    state = np.array(
        initial_state,
        dtype=simulation.ensemble.complex_dtype,
        copy=True,
        order="C",
    )
    expected_shape = (simulation.ensemble.dimension,)
    if state.shape != expected_shape:
        raise ValueError(
            f"`initial_state` must have shape {expected_shape}, got {state.shape}."
        )
    if not np.all(np.isfinite(state)):
        raise ValueError("`initial_state` must contain finite values.")
    norm = float(np.linalg.norm(state))
    tolerance = float(np.sqrt(np.finfo(simulation.ensemble.real_dtype).eps))
    if not np.isclose(norm, 1.0, rtol=tolerance, atol=tolerance):
        raise ValueError(f"`initial_state` must have unit norm, got {norm}.")
    state /= norm
    state.flags.writeable = False
    return state


def run_cdo_evolution(
    ensemble: ManyBodyEnsemble,
    realizs: int,
    *,
    initial_state: np.ndarray | None = None,
    num_times: int = NUM_TIMES,
    logD_time_support: tuple[float, float] = LOGD_TIME_SUPPORT,
    time_chunk_size: int = TIME_CHUNK_SIZE,
    retain_evolved_states: bool = False,
) -> None:
    kwargs: dict[str, Any] = {
        "ensemble": ensemble,
        "realizs": realizs,
        "num_times": num_times,
        "logD_time_support": logD_time_support,
        "time_chunk_size": time_chunk_size,
        "retain_evolved_states": retain_evolved_states,
    }
    if initial_state is not None:
        kwargs["initial_state"] = initial_state
    CDOEvolutionSimulation(**kwargs).run()


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class CDOEvolutionSimulation(Simulation):
    """Monte Carlo evolution of an ensemble-averaged pure-state density operator."""

    ensemble: ManyBodyEnsemble = attrs.field(converter=ManyBodyEnsemble.create)
    realizs: int = attrs.field(
        converter=int,
        validator=attrs.validators.gt(0),
        metadata=REALIZATIONS_METADATA,
    )
    initial_state: np.ndarray = attrs.field(
        default=attrs.Factory(create_default_initial_state, takes_self=True),
        converter=attrs.Converter(normalize_initial_state, takes_self=True),
        repr=False,
    )
    num_times: int = attrs.field(
        default=NUM_TIMES,
        converter=int,
        validator=attrs.validators.gt(0),
    )
    logD_time_support: tuple[float, float] = attrs.field(
        default=LOGD_TIME_SUPPORT,
        converter=tuple,
        validator=lambda _, __, support: rmtpy.validators.validate_support(support),
    )
    time_chunk_size: int = attrs.field(
        default=TIME_CHUNK_SIZE,
        converter=int,
        validator=attrs.validators.gt(0),
        repr=False,
    )
    retain_evolved_states: bool = attrs.field(
        default=False,
        converter=attrs.converters.to_bool,
    )

    outputs: CDOEvolutionOutputs = attrs.field(
        default=attrs.Factory(create_cdo_evolution_outputs, takes_self=True),
        init=False,
        repr=False,
    )

    @property
    def time_scale(self) -> float:
        return FIRST_J1_ZERO / self.ensemble.spectral_radius

    @property
    def to_path(self) -> Path:
        return rmtpy.conversion.to_path(
            self,
            root=Path(self.path_name) / self.ensemble.to_path,
        )

    def realize_monte_carlo_simulation(self) -> None:
        self.outputs.reset()
        times = self.outputs.dynamics.data.times
        dimension = self.ensemble.dimension
        evolved_states = np.empty(
            (self.num_times, dimension),
            dtype=self.ensemble.complex_dtype,
        )
        phase_coefficients = np.empty(
            (min(self.time_chunk_size, self.num_times), dimension),
            dtype=self.ensemble.complex_dtype,
        )

        for eigvals, eigvecs in self.ensemble.eigsys_stream(self.realizs):
            rotated_state = eigvecs.conj().T @ self.initial_state

            for start in range(0, self.num_times, self.time_chunk_size):
                stop = min(start + self.time_chunk_size, self.num_times)
                phase_chunk = phase_coefficients[: stop - start]

                np.multiply(
                    times[start:stop, np.newaxis],
                    eigvals[np.newaxis, :],
                    out=phase_chunk,
                )
                phase_chunk *= -1j
                np.exp(phase_chunk, out=phase_chunk)
                phase_chunk *= rotated_state
                np.matmul(
                    phase_chunk,
                    eigvecs.T,
                    out=evolved_states[start:stop],
                )

            self.outputs.add_evolved_states(evolved_states)

    def calculate_statistics(self) -> None:
        self.outputs.calculate_dynamics()
