# rmtpy/ensembles/gue.py

# Postponed evaluation of annotations
from __future__ import annotations

# Third-party imports
import numpy as np
from attrs import frozen

# Local application imports
from .base import GaussianEnsemble


# -------------------------------
# Gaussian Unitary Ensemble (GUE)
# -------------------------------
@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class GUE(GaussianEnsemble):

    @property
    def beta(self) -> int:
        """Dyson index of the GUE."""
        return 2

    def generate(self, offset: np.ndarray | None = None) -> np.ndarray:
        """Generate a random matrix from the GUE."""

        # If offset is None, allocate memory for matrix
        if offset is None:
            H = np.zeros((self.dim, self.dim), dtype=self.dtype, order="F")
        else:
            H = offset

        # Loop over diagonal indices
        for i in range(self.dim):
            # Generate a random array of standard normal values for real parts
            real_rands = self.rng.standard_normal(self.dim - i, dtype=self.real_dtype)

            # Generate a random array of standard normal values for imaginary parts
            imag_rands = self.rng.standard_normal(self.dim - i, dtype=self.real_dtype)

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
