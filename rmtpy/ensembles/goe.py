# rmtpy/ensembles/goe.py

# Postponed evaluation of annotations
from __future__ import annotations

# Third-party imports
import numpy as np
from attrs import field, frozen

# Local application imports
from ._base import GaussianEnsemble


# ----------------------------------
# Gaussian Orthogonal Ensemble (GOE)
# ----------------------------------
@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class GOE(GaussianEnsemble):

    # Dyson index (for GOE is 1)
    beta: int = field(init=False, default=1, repr=False)

    def generate_matrix(self, offset: np.ndarray | None = None) -> np.ndarray:
        """Generate a random matrix from the GOE."""

        # Alias random number generator
        rng = self.rng

        # Alias dimension of matrix
        d = self.dim

        # Alias data types of matrix elements
        cdtype = self.dtype
        rdtype = self.real_dtype

        # Alias standard deviation of matrix elements
        sigma = self.sigma

        # =============================================================

        # If offset is not None, add to provided matrix
        if offset is not None:
            H = offset

        # Otherwise, create new matrix
        else:
            H = np.zeros((d, d), cdtype, "F")

        # Loop over diagonal indices
        for i in range(d):
            # Generate a random array of standard normal values
            rands = rng.standard_normal(d - i, rdtype)

            # Scale by standard deviation
            rands *= sigma / np.sqrt(2)

            # Add to ith row and ith column of matrix
            H[i, i:].real += rands
            H[i:, i].real += rands

        # Return GOE matrix
        return H
