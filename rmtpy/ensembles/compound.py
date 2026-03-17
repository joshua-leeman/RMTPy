# rmtpy/ensembles/compound.py

# Postponed evaluation of annotations
from __future__ import annotations

# Local application imports
from ._base import ManyBodyEnsemble
from collections.abc import Iterator

# Third-party imports
import numpy as np
from attrs import field, frozen
from attrs.validators import instance_of, gt
from scipy.linalg import eig, eigvals, hadamard, solve


# ---------------------------
# Many-body Compound Ensemble
# ---------------------------
@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class Compound:

    # Underlying ensemble
    ensemble: ManyBodyEnsemble = field(validator=instance_of(ManyBodyEnsemble))

    # Number of open channels
    channels: int = field(converter=int, validator=gt(0))

    # Strength of interaction
    strength: float = field(converter=float, validator=gt(0.0))

    # Couplings matrix
    couplings: np.ndarray = field(init=False, repr=False)

    @couplings.default
    def __couplings_default(self) -> np.ndarray:
        """Create couplings matrix W."""

        # Alias ensemble, its dimension, and its data type
        ensemble = self.ensemble
        dim = ensemble.dim
        dtype = ensemble.dtype

        # Alias number of channels
        M = self.channels

        # Alias strength
        v = self.strength

        # Check if underlying ensemble is time-reversal symmetric
        if ensemble.beta == 1:
            # Select first M columns of Hadamard matrix as couplings
            W = hadamard(dim, dtype=dtype)[:, :M]
        else:
            # Select first M columns of DFT matrix as couplings
            j = np.arange(dim, dtype=dtype)[:, np.newaxis]
            k = np.arange(dim, dtype=dtype)[np.newaxis, :]
            W = np.exp(-2j * np.pi * j * k / dim)[:, :M]

        # Normalize and scale couplings by dimension and strength
        W *= v / np.sqrt(dim)

        # Return couplings matrix
        return np.asfortranarray(W)

    def generate_H_eff(self, out: np.ndarray | None = None) -> np.ndarray:
        """Generate a random effective Hamiltonian."""

        # Alias ensemble
        ensemble = self.ensemble

        # Alias ensemble dimension and data type
        dim = ensemble.dim
        dtype = ensemble.dtype

        # Alias couplings
        W = self.couplings

        # If out is None, allocate memory for effective Hamiltonian
        if out is None:
            H_eff = np.empty((dim, dim), dtype=dtype, order="F")
        else:
            H_eff = out

        # Construct width term of effective Hamiltonian
        H_eff = np.matmul(W, W.conj().T, out=H_eff)
        H_eff *= -1j * np.pi

        # Add random Hamiltonian from underlying ensemble
        ensemble.generate_matrix(offset=H_eff)

        # Return effective Hamiltonian
        return H_eff

    def generate_S(
        self,
        energy: float,
        out: np.ndarray | None = None,
        H_eff: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate a random scattering matrix at given energy."""

        # Alias ensemble dimension and data type
        dim = self.ensemble.dim
        dtype = self.ensemble.dtype

        # Alias number of channels
        M = self.channels

        # Alias couplings and energy
        W = self.couplings
        E = energy

        # If out is None, allocate memory for scattering matrix
        if out is None:
            S = np.empty((M, M), dtype=dtype, order="F")
        else:
            S = out

        # If H_eff is None, generate effective Hamiltonian on the fly
        if H_eff is None:
            H_eff = self.generate_H_eff()

        # Shift effective Hamiltonian by energy
        H_eff -= E * np.eye(dim, dtype=dtype)

        # Compute scattering matrix using the Heidelberg formula
        np.matmul(
            W.conj().T,
            solve(H_eff, W, overwrite_a=True, check_finite=False, assume_a="gen"),
            out=S,
        )
        S *= 2j * np.pi
        S += np.eye(M, dtype=dtype)

        # Return scattering matrix
        return S

    def generate_Q(self, energy: float, out: np.ndarray | None = None) -> np.ndarray:
        pass

    def resonance_stream(self, realizs: int) -> Iterator[np.ndarray]:
        """Iterator to stream effective Hamiltonian eigenvalue realizations (i.e. resonances)."""

        # Alias ensemble dimension and data type
        dim = self.ensemble.dim
        dtype = self.ensemble.dtype

        # Allocate memory for random effective Hamiltonian
        H_eff = np.empty((dim, dim), dtype=dtype, order="F")

        # Loop over realizations
        for _ in range(realizs):
            # Generate effective Hamiltonian
            self.generate_H_eff(out=H_eff)

            # Compute and yield effective Hamiltonian eigenvalues
            yield eigvals(H_eff)

    def S_stream(self, energy: float, realizs: int) -> Iterator[np.ndarray]:
        """Iterator to stream scattering matrix realizations."""

        # Alias ensemble dimension and data type
        dim = self.ensemble.dim
        dtype = self.ensemble.dtype

        # Alias number of channels
        M = self.channels

        # Alias couplings and energy
        W = self.couplings
        E = energy

        # Allocate memory for random effective Hamiltonian and scattering matrix
        H_eff = np.empty((dim, dim), dtype=dtype, order="F")
        S = np.empty((M, M), dtype=dtype, order="F")

        # Loop over realizations
        for _ in range(realizs):
            # Generate effective Hamiltonian
            self.generate_H_eff(offset=H_eff)

            # Shift effective Hamiltonian by energy
            H_eff -= E * np.eye(dim, dtype=dtype, order="F")

            # Compute scattering matrix using the Heidelberg formula
            np.matmul(W.conj().T, np.linalg.solve(H_eff, W), out=S)
            S *= 2j * np.pi
            S += np.eye(M, dtype=dtype)

            # Yield scattering matrix realization
            yield S

    def Q_stream(self, energy: float, realizs: int) -> Iterator[np.ndarray]:
        """Iterator to stream Wigner-Smith time-delay matrix realizations."""

        # Alias ensemble dimension and data type
        dim = self.ensemble.dim
        dtype = self.ensemble.dtype

        # Alias number of channels
        M = self.channels

        # Alias couplings and energy
        W = self.couplings
        E = energy

        # Allocate memory for random effective Hamiltonian and time-delay matrix
        H_eff = np.empty((dim, dim), dtype=dtype, order="F")
        Q = np.empty((M, M), dtype=dtype, order="F")

        # Loop over realizations
        for _ in range(realizs):
            # Generate effective Hamiltonian
            self.generate_H_eff(out=H_eff)

            # Shift effective Hamiltonian by energy
            H_eff -= E * np.eye(dim, dtype=dtype)

            # Compute time-delay matrix
            Y = np.linalg.solve(H_eff, W)
            np.matmul(Y.conj().T, Y, out=Q)
            Q *= 2 * np.pi

            # Yield time-delay matrix realization
            yield Q
