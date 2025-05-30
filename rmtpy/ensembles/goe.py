# rmtpy.ensembles.goe.py


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
class_name = "GOE"


# Define Gaussian Orthogonal Ensemble (GOE) class
@dataclass(repr=False, eq=False, frozen=True, kw_only=True, slots=True)
class GOE(GaussianEnsemble):
    """Gaussian Orthogonal Ensemble (GOE) class."""

    # Dyson index
    beta: int = field(init=False, default=1)

    def randm(self, offset: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate a random matrix from the GOE."""
        # If offset is None, allocate memory for matrix
        if offset is None:
            H = np.zeros((self.dim, self.dim), dtype=self.dtype, order="F")
        else:
            H = offset

        # Loop over diagonal indices
        for i in range(self.dim):
            # Generate a random array of standard normal values
            rands = self._rng.standard_normal(self.dim - i, dtype=self.real_dtype)

            # Scale random values by real standard deviation
            rands *= self.sigma / np.sqrt(2.0)

            # Add to ith row and ith column
            H[i, i:] += rands
            H[i:, i] += rands

        # Return GOE matrix
        return H
