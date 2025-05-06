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

    def randm(self, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate a random matrix from the BdGD."""
        # If out is None, allocate memory for matrix
        if out is None:
            H = np.empty((self.dim, self.dim), dtype=self.dtype, order="F")
        else:
            H = out

        # Generate standard normalss for imaginary parts and zeros for real parts
        H.real = np.zeros(H.shape, dtype=self.real_dtype)
        H.imag = self._rng.standard_normal(H.shape, dtype=self.real_dtype)

        # Anti-symmetrize matrix in place
        np.subtract(H, H.T, out=H)

        # Halve and scale matrix by real standard deviation in place
        H *= self.sigma / 2

        # Return BdG(D) matrix
        return H
