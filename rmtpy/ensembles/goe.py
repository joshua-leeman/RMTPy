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

    def randm(self, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate a random matrix from the GOE."""
        # If out is None, allocate memory for matrix
        if out is None:
            H = np.empty((self.dim, self.dim), dtype=self.dtype, order="F")
        else:
            H = out

        # Set standard normals for real parts and zero for imaginary parts
        H.real = self._rng.standard_normal(H.shape, dtype=self.real_dtype)
        H.imag = np.zeros(H.shape, dtype=self.real_dtype)

        # Symmetrize matrix in place
        np.add(H, H.T, out=H)

        # Halve and scale matrix by real standard deviation in place
        H *= self.sigma / 2

        # Return GOE matrix
        return H
