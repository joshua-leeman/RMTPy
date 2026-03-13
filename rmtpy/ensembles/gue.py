# rmtpy/ensembles/gue.py

# Postponed evaluation of annotations
from __future__ import annotations

# Third-party imports
import numpy as np
from attrs import field, frozen

# Local application imports
from ._base import GaussianEnsemble


# -------------------------------
# Gaussian Unitary Ensemble (GUE)
# -------------------------------
@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class GUE(GaussianEnsemble):

    # Dyson index (for GUE is 2)
    beta: int = field(init=False, default=2, repr=False)

    def generate_matrix(self, offset: np.ndarray | None = None) -> np.ndarray:
        """Generate a random matrix from the GUE."""

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
            # Generate a random array of standard normal values for real parts
            real_rands = rng.standard_normal(d - i, rdtype)

            # Generate a random array of standard normal values for imaginary parts
            imag_rands = rng.standard_normal(d - i, rdtype)

            # Scale by complex standard deviation
            real_rands *= sigma / 2
            imag_rands *= sigma / 2

            # Add to ith row and ith column of matrix
            H[i, i:].real += real_rands
            H[i:, i].real += real_rands
            H[i, i:].imag += imag_rands
            H[i:, i].imag -= imag_rands

        # Return GUE matrix
        return H
