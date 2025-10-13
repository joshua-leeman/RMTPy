# rmtpy/ensembles/goe.py

# Postponed evaluation of annotations
from __future__ import annotations

# Third-party imports
import numpy as np
from attrs import frozen

# Local application imports
from ._base import GaussianEnsemble


# ----------------------------------
# Gaussian Orthogonal Ensemble (GOE)
# ----------------------------------
@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class GOE(GaussianEnsemble):

    @property
    def beta(self) -> int:
        """Dyson index of the GOE."""

        # Ensemble GOE has Dyson index 1
        return 1

    def generate(self, offset: np.ndarray | None = None) -> np.ndarray:
        """Generate a random matrix from the GOE."""

        # If offset is None, allocate memory for matrix
        if offset is None:
            H = np.zeros((self.dim, self.dim), dtype=self.dtype, order="F")
        else:
            H = offset

        # Loop over diagonal indices
        for i in range(self.dim):
            # Generate a random array of standard normal values
            rands = self.rng.standard_normal(self.dim - i, dtype=self.real_dtype)

            # Scale random values by real standard deviation
            rands *= self.sigma / np.sqrt(2.0)

            # Add to ith row and ith column
            H[i, i:].real += rands
            H[i:, i].real += rands

        # Return GOE matrix
        return H
