# rmtpy/simulations/evolve_cdo_simulation.py

# Postponed evaluation of annotations
from __future__ import annotations

# Standard library imports
from pathlib import Path

# Third-party imports
import numpy as np
from attrs import frozen, field
from scipy.linalg import eigvalsh
from scipy.special import jn_zeros

# Local application imports
from ._simulation import Simulation
from ..data.evolve_cdo_data import CDODynamicsData, EvolvedStatesData
from ..ensembles import ManyBodyEnsemble


# ---------------------------------
# CDO Evolution Simulation Function
# ---------------------------------


# ------------------------------
# CDO Evolution Simulation Class
# ------------------------------
@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class CDOEvolution(Simulation):

    # Initial state
    initial_state: np.ndarray = field(repr=False)

    # CDO dynamics data
    cdo_dynamics: CDODynamicsData = field(init=False, repr=False)

    # Evolved states data
    evolved_states: EvolvedStatesData = field(repr=False)

    @initial_state.default
    def __initial_state_default(self) -> np.ndarray:
        """Initialize the initial state vector."""

        # Alias ensemble dimension and dtype
        dim: int = self.ensemble.dim
        dtype: np.dtype = self.ensemble.dtype

        # Set initial state to first basis vector
        state = np.zeros(dim, dtype=dtype, order="F")
        state[0] = 1.0

        # Return initial state
        return state

    @cdo_dynamics.default
    def __cdo_dynamics_default(self) -> CDODynamicsData:
        """Initialize the CDO dynamics data."""

        # Return CDO dynamics data instance
        return CDODynamicsData(dim=self.ensemble.dim)

    @evolved_states.default
    def __evolved_states_default(self) -> EvolvedStatesData:
        """Initialize the evolved states data."""

        # Return evolved states data instance
        return EvolvedStatesData(
            realizs=self.realizs,
            num_times=self.cdo_dynamics.num_times,
            dim=self.ensemble.dim,
            dtype=self.ensemble.dtype,
        )

    def realize_monte_carlo(self) -> None:
        """Realize Monte Carlo sample of the CDO evolution."""

        # Retrieve first positive zero of 1st Bessel function
        j_1_1 = jn_zeros(1, 1)[0]

        # Alias ensemble
        ensemble: ManyBodyEnsemble = self.ensemble

        # Alias ensemble dimension
        E0: float = ensemble.E0

        # Alias number of realizations
        realizs: int = self.realizs

        # Alias initial state
        initial_state: np.ndarray = self.initial_state

        # Alias evolved states data
        evolved_states: np.ndarray = self.evolved_states.states

        # Alias times
        times: np.ndarray = self.cdo_dynamics.times

        # Change base of times to dimension
        times **= np.log(ensemble.dim) / np.log(10.0)

        # Scale times by j_1_1 / E0
        times *= j_1_1 / E0

        # Loop over diagonalization realizations
        for r, eigsys in enumerate(ensemble.eig_stream(realizs)):

            # Unpack eigenvalues and eigenvectors
            eigvals, eigvecs = eigsys

            # Rotate initial state into energy eigenbasis
            rotated_state = np.matmul(eigvecs.T.conj(), initial_state)

            # Outer-multiply eigenvalues and times
            np.outer(times, eigvals, out=evolved_states[r])

            # Complex exponentiate
            evolved_states[r] *= -1j
            np.exp(evolved_states[r], out=evolved_states[r])

            # Broadcast multiply by rotated initial state
            np.multiply(evolved_states[r], rotated_state, out=evolved_states[r])

            # Rotate evolved states back to original basis
            np.matmul(evolved_states[r], eigvecs.T, out=evolved_states[r])

    def calculate_cdo_dynamics(self) -> None:
        """Calculate CDO dynamics from evolved states."""

        # Alias ensemble dimension and dtype
        dim: int = self.ensemble.dim
        dtype: np.dtype = self.ensemble.dtype

        # Alias data array of evolved states
        states: np.ndarray = self.evolved_states.states

        # Transpose evolved states for easier calculations
        states = states.transpose(1, 0, 2)

        # Alias number of realizations and times
        realizs: int = self.evolved_states.realizs
        num_times: int = self.evolved_states.num_times

        # Initialize memory for storing CDOs
        cdo = np.empty((dim, dim), dtype=dtype, order="F")

        # Loop through times to calculate CDO dynamics
        for t in range(num_times):

            # Calculate chaotic density operator
            cdo = np.matmul(states[t].conj().T, states[t], out=cdo)
            cdo /= realizs

            # Calculate probabilities
            cdo_diag = np.diagonal(cdo).real
            self.cdo_dynamics.probs[t, :] = cdo_diag

            # Calculate classical purity
            self.cdo_dynamics.c_purity[t] = np.sum(cdo_diag**2)

            # Compute eigenvalues of CDO
            eigvals = eigvalsh(cdo, overwrite_a=True, check_finite=False)

            # Calculate quantum purity
            self.cdo_dynamics.q_purity[t] = np.sum(eigvals**2)

            # Calculate von Neumann entropy
            self.cdo_dynamics.entropy[t] = -np.sum(eigvals * np.log(eigvals))
