# rmtpy/compounds/_compound.py

# Postponed evaluation of annotations
from __future__ import annotations

# Standard library imports
import math
from abc import ABC, abstractmethod

# Third-party imports
import numpy as np
from attrs import Converter, field, frozen
from attrs.validators import instance_of
from scipy.linalg import eigvals, solve

# Local application imports
from ..ensembles import ManyBodyEnsemble
from collections.abc import Iterator


# ------------------------
# Energies Array Converter
# ------------------------
def _to_1D_array(energies: float | np.ndarray) -> np.ndarray:
    """Ensure energies is a 1D array with real values."""

    # If energies is not a float or an array, raise an error
    if not isinstance(energies, (float, np.ndarray)):
        raise TypeError(
            f"Energies must be a float or a numpy array, got {type(energies).__name__} instead."
        )

    # If energies is a float, convert it to a 1D array with one element
    elif isinstance(energies, float):
        return np.array([energies], dtype=np.float64, order="F")

    # Else ensure it is 1D and has real values
    else:
        if energies.ndim != 1:
            raise ValueError(
                f"Energies must be a 1D array, got {energies.ndim}D array."
            )
        elif not np.isrealobj(energies):
            raise ValueError("Energies must have real values.")

        # Return energies as a 1D array
        return energies.astype(np.float64)


# -----------------------------
# Couplings Strengths Converter
# -----------------------------
def _to_full_array(value: float | np.ndarray, self_: Compound) -> np.ndarray:
    """If value is a scalar, create an array of coupling strengths with the same value for all entries."""

    # Alias number of open channels
    channels = self_.channels

    # =================================================

    # If value is a scalar, set all couplings to the same value
    if isinstance(value, (int, float)):
        return np.full(channels, value)

    # If value is an array, ensure it has the correct shape
    elif isinstance(value, np.ndarray):
        if value.shape != (channels,):
            raise ValueError(
                f"Coupling strengths array must have shape ({channels},), got {value.shape}."
            )

        # Return array of coupling strengths in Fortran order
        return np.asfortranarray(value)

    # If value is neither a scalar nor an array, raise an error
    else:
        raise TypeError(
            f"Coupling strengths must be a scalar or a numpy array, got {type(value).__name__}."
        )


# ---------------------------
# Many-body Compound Ensemble
# ---------------------------
@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class Compound(ABC):

    # Underlying RMT ensemble
    ensemble: ManyBodyEnsemble = field(validator=instance_of(ManyBodyEnsemble))

    # Number of asymptotically-free fermions
    fermions: int = field(default=1, converter=int)

    # Number of open channels
    channels: int = field(init=False)

    # Coupling strengths
    strengths: np.ndarray = field(converter=Converter(_to_full_array, takes_self=True))

    @fermions.validator
    def __fermions_validator(self, _, value: int) -> None:
        """Ensure number of fermions is a nonnegative integer."""

        # Alias number of complex fermions in ensemble
        Nc = self.ensemble.Nm // 2

        # =================================================

        # Ensure number of fermions is nonnegative and less than or equal to Nc
        if value < 0 or value > Nc:
            raise ValueError(
                f"Number of fermions must be a nonnegative integer less than or equal to {Nc}, got {value}."
            )

    @channels.default
    def __channels_default(self) -> int:
        """Set number of open channels equal to number of combinations of complex fermions from the ensemble."""

        # Alias number of asymptotically-free fermions
        k = self.fermions

        # Alias number of complex fermions in ensemble
        Nc = self.ensemble.Nm // 2

        # =================================================

        # Calculate and return number of combinations
        return math.comb(Nc, k)

    @strengths.default
    def __strengths_default(self) -> float:
        """Set uniform strengths to square root of ground state energy."""

        # Alias ensemble ground state energy
        E0 = self.ensemble.E0

        # =================================================

        # Set uniform strengths to square root of ground state energy
        return math.sqrt(E0)

    def generate_H_eff(
        self, out: np.ndarray | None = None, offset: np.ndarray | None = None
    ) -> np.ndarray:
        """Generate a random effective Hamiltonian."""

        # Alias ensemble
        ensemble = self.ensemble

        # Alias complex data type of ensemble
        cdtype = ensemble.dtype.type

        # Alias dimension of ensemble matrices
        d = ensemble.dim

        # =================================================

        # If offset is not None, add to provided matrix
        if offset is not None:
            # Alias provided matrix
            H_eff = offset

            # Add random matrix from ensemble to H_eff
            ensemble.generate_matrix(offset=H_eff)

        # Otherwise, write to provided memory
        else:
            # Alias memory for output matrix
            if out is not None:
                # Alias provided matrix
                H_eff = out
            else:
                # Create empty matrix
                H_eff = np.empty((d, d), cdtype, order="F")

            # Create random Hamiltonian
            ensemble.generate_matrix(out=H_eff)

        # Add width matrix contribution to H_eff
        self.generate_width_matrix(offset=H_eff)

        # Return effective Hamiltonian
        return H_eff

    def resonance_stream(self, realizs: int) -> Iterator[np.ndarray]:
        """Iterator to stream effective Hamiltonian eigenvalues (resonances) realizations."""

        # Alias ensemble dimension
        dim = self.ensemble.dim

        # Alias ensemble data type
        dtype = self.ensemble.dtype

        # =================================================

        # Allocate memory for random effective Hamiltonians
        H_eff = np.empty((dim, dim), dtype=dtype, order="F")

        # For each realization . . .
        for _ in range(realizs):
            # Generate effective Hamiltonian
            self.generate_H_eff(out=H_eff)

            # Compute and yield effective Hamiltonian eigenvalues
            yield eigvals(H_eff, overwrite_a=True, check_finite=False)

    def S_matrix_stream(
        self, energies: float | np.ndarray, realizs: int
    ) -> Iterator[np.ndarray]:
        """Iterator to stream S matrix realizations at given energies."""

        # Ensure energies is a 1D array
        energies = _to_1D_array(energies)

        # Alias number of energies
        num = energies.size

        # Alias ensemble data type
        dtype = self.ensemble.dtype

        # Alias number of open channels
        L = self.channels

        # =================================================

        # Allocate memory for coefficient matrix conjugate
        Cc = np.empty((num, L, L), dtype=dtype, order="C")

        # For K-matrix realizations . . .
        for C in self.K_matrix_stream(energies, realizs):

            # Calculate coefficient matrix
            C *= 1j * np.pi
            C.diagonal(axis1=-2, axis2=-1)[:] += 1

            # Store conjugate of coefficient matrix in Cc
            np.conjugate(C, out=Cc)

            # Yield S-matrix realization
            yield solve(C, Cc, overwrite_a=True, overwrite_b=True, check_finite=False)

    def Q_matrix_stream(
        self, energies: float | np.ndarray, realizs: int
    ) -> Iterator[np.ndarray]:
        """Iterator to stream Q matrix realizations at given energies."""

        # Ensure energies is a 1D array
        energies = _to_1D_array(energies)

        # =================================================

        # For K, K_2 matrix realizations . . .
        for C, pi_K_2 in self.K_K2_matrix_stream(energies, realizs):

            # Calculate coefficient matrix
            C *= -1j * np.pi
            C.diagonal(axis1=-2, axis2=-1)[:] += 1

            # Scale K_2 matrix by pi
            pi_K_2 *= np.pi

            # Calculate Q-matrix realization
            Q = solve(C, pi_K_2, overwrite_a=True, overwrite_b=True, check_finite=False)

            # Add its conjugate transpose to itself in place
            Q += Q.swapaxes(-1, -2).conj()

            # Yield Q-matrix realization
            yield Q

    def time_delays_stream(
        self, energies: float | np.ndarray, realizs: int
    ) -> Iterator[np.ndarray]:
        """Iterator to stream time delay realizations at given energies."""

        # Ensure energies is a 1D array
        energies = _to_1D_array(energies)

        # =================================================

        # For Q-matrix realizations . . .
        for Q in self.Q_matrix_stream(energies, realizs):

            # Batch-compute eigenvalues of Q-matrix realization and yield
            yield np.linalg.eigvalsh(Q)

    @abstractmethod
    def generate_width_matrix(
        self, out: np.ndarray | None = None, offset: np.ndarray | None = None
    ) -> np.ndarray:
        """Generate the width matrix."""

        # This method should be implemented by subclasses
        pass

    @abstractmethod
    def partial_widths_stream(self, realizs: int) -> Iterator[np.ndarray]:
        """Iterator to stream partial widths realizations."""

        # This method should be implemented by subclasses
        pass

    @abstractmethod
    def K_matrix_stream(
        self, energies: float | np.ndarray, realizs: int
    ) -> Iterator[np.ndarray]:
        """Iterator to stream K matrix realizations at energies."""

        # This method should be implemented by subclasses
        pass

    @abstractmethod
    def K_K2_matrix_stream(
        self, energies: float | np.ndarray, realizs: int
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Iterator to stream K and K_2 matrix realizations at energies."""

        # This method should be implemented by subclasses
        pass
