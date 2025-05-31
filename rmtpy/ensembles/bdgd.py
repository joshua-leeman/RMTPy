# rmtpy.ensembles.bdgd.py


# =======================================
# 1. Imports
# =======================================
# Standard library imports
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

# Third-party imports
import numpy as np

# Local application imports
from rmtpy.ensembles._rmt import GaussianEnsemble


# =======================================
# 2. Ensemble
# =======================================
# Store class name for module
class_name = "BdGD"


# Define Bogoliubov-de Gennes D Ensemble (BdG(D)) class
@dataclass(repr=False, eq=False, frozen=True, kw_only=True, slots=True)
class BdGD(GaussianEnsemble):
    """Bogoliubov-de Gennes D Ensemble (BdG(D)) class."""

    # Dyson index
    beta: int = field(init=False, default=2)

    def _to_latex(self) -> str:
        """LaTeX representation of the ensemble."""
        # Return formatted LaTeX string
        return rf"$\textrm{{BdG(D)}}\ N={self.N}$"

    def randm(self, offset: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate a random matrix from the BdGD."""
        # If out is None, allocate memory for matrix
        if offset is None:
            H = np.zeros((self.dim, self.dim), dtype=self.dtype, order="F")
        else:
            H = offset

        # Loop over diagonal indices
        for i in range(self.dim):
            # Generate random array of standard normal values for imaginary parts
            imag_rands = np.zeros(self.dim - i, dtype=self.real_dtype)
            imag_rands[1:] += self._rng.standard_normal(
                self.dim - i - 1, dtype=self.real_dtype
            )

            # Scale random values by complex standard deviation
            imag_rands *= self.sigma / np.sqrt(2.0)

            # Add imaginary parts to the ith row and column
            H[i, i:].imag += imag_rands
            H[i:, i].imag -= imag_rands

        # Return BdG(D) matrix
        return H
