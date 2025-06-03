# rmtpy.ensembles.compound.py


# =======================================
# 1. Imports
# =======================================
# Standard library imports
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

# Third-party imports
import numpy as np
from scipy.linalg import eig, eigvals, hadamard

# Local application imports
from rmtpy.ensembles._rmt import ManyBodyEnsemble


# =======================================
# 2. Many-body Compound Ensemble
# =======================================
# Store class name for module
class_name = "Compound"


# Define Many-body Compound Ensemble class
@dataclass(repr=False, eq=False, frozen=False, kw_only=True, slots=True)
class Compound:
    # Underlying ensemble
    ensemble: ManyBodyEnsemble

    # Number of open channels
    channels: int = 1

    # Strength of interaction
    strength: float = 1.0

    # Time reversal symmetry flag
    time_reversal: bool = False

    # Coupling matrix
    coupling: Optional[np.ndarray] = field(init=False, default=None)

    def __post_init__(self):
        # Check if ensemble is an instance of ManyBodyEnsemble
        if not isinstance(self.ensemble, ManyBodyEnsemble):
            raise TypeError("ensemble must be an instance of ManyBodyEnsemble")

        # Check if channels is a positive integer
        if not isinstance(self.channels, int) or self.channels <= 0:
            raise ValueError("channels must be a positive integer")

        # Check if strength is a positive float
        if not isinstance(self.strength, (float, int)) or self.strength <= 0:
            raise ValueError("strength must be a positive float or int")

        # Create coupling matrix
        self.create_couplings()

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying ensemble."""
        if hasattr(self.ensemble, name):
            return getattr(self.ensemble, name)
        raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")

    def create_couplings(self) -> np.ndarray:
        # If TRS enabled, select columns from a Hadamard matrix
        if self.time_reversal:
            # Select columns from Hadamard matrix of size ensemble dimension
            W = hadamard(self.dim, dtype=self.dtype)[:, : self.channels]

            # Normalize columns to have norms equal to strength in place
            W *= self.strength / np.sqrt(self.dim)

            # Store coupling matrix
            object.__setattr__(self, "coupling", W)
        # If TRS disabled, create normalized DFT matrix
        else:
            # Create only first columns from normalized DFT matrix
            j = np.arange(self.dim, dtype=self.dtype)[:, np.newaxis]
            k = np.arange(self.channels, dtype=self.dtype)[np.newaxis, :]
            W = np.exp(-2j * np.pi * j * k / self.dim)

            # Normalize columns to have norms equal to strength in place
            W *= self.strength / np.sqrt(self.dim)

            # Store coupling matrix
            object.__setattr__(self, "coupling", W)

    def eff_H_eig_stream(self, realizs: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        # Allocate memory for random effective Hamiltonian stream
        H_eff = np.empty((self.dim, self.dim), dtype=self.dtype)

        # Loop over realizations
        for _ in range(realizs):
            # Reset effective Hamiltonian with width term
            np.matmul(self.coupling, self.coupling.conj().T, out=H_eff)
            H_eff *= -1j * np.pi

            # Add random part from underlying ensemble
            self.ensemble.randm(offset=H_eff)

            # Compute and yield eigenvalues
            yield eig(H_eff, overwrite_a=True, check_finite=False)

    def eff_H_eigvals_stream(self, realizs: int) -> Iterator[np.ndarray]:
        # Allocate memory for random effective Hamiltonian stream
        H_eff = np.empty((self.dim, self.dim), dtype=self.dtype)

        # Loop over realizations
        for _ in range(realizs):
            # Reset effective Hamiltonian with width term
            np.matmul(self.coupling, self.coupling.conj().T, out=H_eff)
            H_eff *= -1j * np.pi

            # Add random part from underlying ensemble
            self.ensemble.randm(offset=H_eff)

            # Compute and yield eigenvalues
            yield eigvals(H_eff, overwrite_a=True, check_finite=False)
