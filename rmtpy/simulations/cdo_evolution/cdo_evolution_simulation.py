from __future__ import annotations

from pathlib import Path

import numpy as np
from attrs import frozen, field
from scipy.linalg import eigvalsh
from scipy.special import jn_zeros

from .._simulation import Simulation
from .cdo_evolution_data import CDODynamicsData, EvolvedStatesData
from ...ensembles import ManyBodyEnsemble


@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class CDOEvolution(Simulation):

    initial_state: np.ndarray = field(repr=False)

    cdo_dynamics: CDODynamicsData = field(init=False, repr=False)

    evolved_states: EvolvedStatesData = field(repr=False)

    @initial_state.default
    def __initial_state_default(self) -> np.ndarray:

        dim: int = self.ensemble.dim
        dtype: np.dtype = self.ensemble.dtype

        state = np.zeros(dim, dtype=dtype, order="F")
        state[0] = 1.0

        return state

    @cdo_dynamics.default
    def __cdo_dynamics_default(self) -> CDODynamicsData:
        return CDODynamicsData(dim=self.ensemble.dim)

    @evolved_states.default
    def __evolved_states_default(self) -> EvolvedStatesData:

        return EvolvedStatesData(
            realizs=self.realizs,
            num_times=self.cdo_dynamics.num_times,
            dim=self.ensemble.dim,
            dtype=self.ensemble.dtype,
        )

    def realize_monte_carlo(self) -> None:

        j_1_1 = jn_zeros(1, 1)[0]

        ensemble: ManyBodyEnsemble = self.ensemble

        E0: float = ensemble.E0

        realizs: int = self.realizs

        initial_state: np.ndarray = self.initial_state

        evolved_states: np.ndarray = self.evolved_states.states

        times: np.ndarray = self.cdo_dynamics.times

        times **= np.log(ensemble.dim) / np.log(10.0)

        times *= j_1_1 / E0

        for r, eigsys in enumerate(ensemble.eigsys_stream(realizs)):

            eigvals, eigvecs = eigsys

            rotated_state = np.matmul(eigvecs.T.conj(), initial_state)

            np.outer(times, eigvals, out=evolved_states[r])

            evolved_states[r] *= -1j
            np.exp(evolved_states[r], out=evolved_states[r])

            np.multiply(evolved_states[r], rotated_state, out=evolved_states[r])

            np.matmul(evolved_states[r], eigvecs.T, out=evolved_states[r])

    def calculate_cdo_dynamics(self) -> None:

        dim: int = self.ensemble.dim
        dtype: np.dtype = self.ensemble.dtype

        states: np.ndarray = self.evolved_states.states

        states = states.transpose(1, 0, 2)

        realizs: int = self.evolved_states.realizs
        num_times: int = self.evolved_states.num_times

        cdo = np.empty((dim, dim), dtype=dtype, order="F")

        for t in range(num_times):

            cdo = np.matmul(states[t].conj().T, states[t], out=cdo)
            cdo /= realizs

            cdo_diag = np.diagonal(cdo).real
            self.cdo_dynamics.probs[t, :] = cdo_diag

            self.cdo_dynamics.c_purity[t] = np.sum(cdo_diag**2)

            eigvals = eigvalsh(cdo, overwrite_a=True, check_finite=False)

            self.cdo_dynamics.q_purity[t] = np.sum(eigvals**2)

            self.cdo_dynamics.entropy[t] = -np.sum(eigvals * np.log(eigvals))
