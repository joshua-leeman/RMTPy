# rmtpy.ensembles.gue.py


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
class_name = "GUE"


# Define Gaussian Unitary Ensemble (GUE) class
@dataclass(repr=False, eq=False, frozen=True, kw_only=True, slots=True)
class GUE(GaussianEnsemble):
    """Gaussian Unitary Ensemble (GUE) class."""

    # Dyson index
    beta: int = field(init=False, default=2)

    def randm(self, offset: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate a random matrix from the GUE."""
        # If offset is None, allocate memory for matrix
        if offset is None:
            H = np.zeros((self.dim, self.dim), dtype=self.dtype, order="F")
        else:
            H = offset

        # Loop over diagonal indices
        for i in range(self.dim):
            # Generate a random array of standard normal values for real parts
            real_rands = self._rng.standard_normal(self.dim - i, dtype=self.real_dtype)

            # Generate a random array of standard normal values for imaginary parts
            imag_rands = self._rng.standard_normal(self.dim - i, dtype=self.real_dtype)

            # Scale random values by complex standard deviation
            real_rands *= self.sigma / 2
            imag_rands *= self.sigma / 2

            # Add real and imaginary parts to the ith row and column
            H[i, i:].real += real_rands
            H[i, i:].imag += imag_rands
            H[i:, i].real += real_rands
            H[i:, i].imag -= imag_rands

        # Return GUE matrix
        return H
