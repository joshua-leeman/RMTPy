# rmtpy/ensembles/gse.py

# Postponed evaluation of annotations
from __future__ import annotations

# Third-party imports
import numpy as np
from attrs import frozen

# Local application imports
from ._base import GaussianEnsemble


# ----------------------------------
# Gaussian Symplectic Ensemble (GSE)
# ----------------------------------
@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class GSE(GaussianEnsemble):

    @property
    def beta(self) -> int:
        """Dyson index of the GSE."""
        return 4

    def generate(self, offset: np.ndarray | None = None) -> np.ndarray:
        """Generate a random matrix from the GSE."""

        # If offset is None, allocate memory for matrix
        if offset is None:
            H = np.zeros((self.dim, self.dim), dtype=self.dtype, order="F")
        else:
            H = offset

        # Compute block dimension
        bdim = self.dim // 2

        # Generate blocks of GSE matrix
        for i in range(bdim):
            # Generate a random array of standard normal values for real parts
            real_rands = self.rng.standard_normal(bdim - i, dtype=self.real_dtype)

            # Generate a random array of standard normal values for imaginary parts
            imag_rands = self.rng.standard_normal(bdim - i, dtype=self.real_dtype)

            # Scale random values by complex standard deviation
            real_rands *= self.sigma / 2
            imag_rands *= self.sigma / 2

            # Add real and imaginary parts of GUE in top-left block
            H[i, i:bdim].real += real_rands
            H[i, i:bdim].imag += imag_rands
            H[i:bdim, i].real += real_rands
            H[i:bdim, i].imag -= imag_rands

            # Add conjugate of top-left block to bottom-right block
            H[bdim + i, bdim + i :].real += real_rands
            H[bdim + i, bdim + i :].imag -= imag_rands
            H[bdim + i :, bdim + i].real += real_rands
            H[bdim + i :, bdim + i].imag += imag_rands

            # Generate random array of standard normal values for real parts
            real_rands = np.zeros(bdim - i, dtype=self.real_dtype)
            real_rands[1:] += self.rng.standard_normal(
                bdim - i - 1, dtype=self.real_dtype
            )

            # Generate random array of standard normal values for imaginary parts
            imag_rands = np.zeros(bdim - i, dtype=self.real_dtype)
            imag_rands[1:] += self.rng.standard_normal(
                bdim - i - 1, dtype=self.real_dtype
            )

            # Scale random values by complex standard deviation
            real_rands *= self.sigma / 2
            imag_rands *= self.sigma / 2

            # Add real and imaginary parts of complex anti-symmetric in top-right block
            H[i, bdim + i :].real += real_rands
            H[i, bdim + i :].imag += imag_rands
            H[i:bdim, bdim + i].real -= real_rands
            H[i:bdim, bdim + i].imag -= imag_rands

            # Add negative conjugate of top-right block to bottom-left block
            H[bdim + i, i:bdim].real -= real_rands
            H[bdim + i, i:bdim].imag += imag_rands
            H[bdim + i :, i].real += real_rands
            H[bdim + i :, i].imag -= imag_rands

        # Return GSE matrix
        return H
