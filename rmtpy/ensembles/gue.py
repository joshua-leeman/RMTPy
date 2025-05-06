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

    def randm(self, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate a random matrix from the GUE."""
        # If out is None, allocate memory for matrix
        if out is None:
            H = np.empty((self.dim, self.dim), dtype=self.dtype, order="F")
        else:
            H = out

        # Generate standard normal numbers for real and imaginary parts
        H.real = self._rng.standard_normal(H.shape, dtype=self.real_dtype)
        H.imag = self._rng.standard_normal(H.shape, dtype=self.real_dtype)

        # Adjoint matrix in place
        np.add(H, H.T.conj(), out=H)

        # Halve and scale matrix by complex standard deviation in place
        H *= self.sigma / np.sqrt(2) / 2

        # Return GUE matrix
        return H
