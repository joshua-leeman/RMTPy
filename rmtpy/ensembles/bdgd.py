# rmtpy/ensembles/bdgd.py

# Postponed evaluation of annotations
from __future__ import annotations

# Third-party imports
import numpy as np
from attrs import frozen

# Local imports
from .base.gaussian import GaussianEnsemble


# ----------------------------------------
# Bogoliubov-de Gennes D Ensemble (BdG(D))
# ----------------------------------------
@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class BdGD(GaussianEnsemble):
    @property
    def beta(self) -> int:
        """Dyson index of the BdG(D)."""
        return 2

    def generate(self, offset: np.ndarray | None = None) -> np.ndarray:
        """Generate a random matrix from the BdG(D) ensemble."""
        # If out is None, allocate memory for matrix
        if offset is None:
            H = np.zeros((self.dim, self.dim), dtype=self.dtype, order="F")
        else:
            H = offset

        # Loop over diagonal indices
        for i in range(self.dim):
            # Generate random array of standard normal values for imaginary parts
            imag_rands = np.zeros(self.dim - i, dtype=self.real_dtype)
            imag_rands[1:] += self.rng.standard_normal(
                self.dim - i - 1, dtype=self.real_dtype
            )

            # Scale random values by complex standard deviation
            imag_rands *= self.sigma / np.sqrt(2.0)

            # Add imaginary parts to the ith row and column
            H[i, i:].imag += imag_rands
            H[i:, i].imag -= imag_rands

        # Return BdG(D) matrix
        return H
