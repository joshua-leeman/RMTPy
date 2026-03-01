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
from scipy.linalg import eig, eigvals, hadamard


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
        return W

    def H_eff(self, out: np.ndarray | None = None) -> np.ndarray:
        """Generate a random effective Hamiltonian."""

        # Alias ensemble
        ensemble = self.ensemble

        # Alias couplings
        W = self.couplings

        # Construct width term of effective Hamiltonian
        H_eff = np.matmul(W, W.conj().T, out=out)
        H_eff *= -1j * np.pi

        # Add random Hamiltonian from underlying ensemble
        ensemble.generate(H_eff, offset=H_eff)

        # Return effective Hamiltonian
        return H_eff

    def resonance_stream(self, realizs: int) -> Iterator[np.ndarray]:
        """Iterator to stream resonances."""

        # Alias ensemble dimension and data type
        dim = self.ensemble.dim
        dtype = self.ensemble.dtype

        # Allocate memory for random effective Hamiltonian
        H_eff = np.empty((dim, dim), dtype=dtype)

        # Loop over realizations
        for _ in range(realizs):
            # Generate effective Hamiltonian
            self.H_eff(out=H_eff)

            # Compute and yield resonance eigenvalues
            yield eigvals(H_eff)

    def eff_H_eigsys_stream(
        self, realizs: int
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Iterator to stream effective Hamiltonian eigensystem realizations."""

        # Alias ensemble dimension and data type
        dim = self.ensemble.dim
        dtype = self.ensemble.dtype

        # Allocate memory for random effective Hamiltonian
        H_eff = np.empty((dim, dim), dtype=dtype)

        # Loop over realizations
        for _ in range(realizs):
            # Generate effective Hamiltonian
            self.H_eff(out=H_eff)

            # Compute and yield eigenvalues and eigenvectors
            yield eig(H_eff, overwrite_a=True, check_finite=False)
