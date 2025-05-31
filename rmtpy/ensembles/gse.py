# rmtpy.ensembles.gse.py


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
class_name = "GSE"


# Define Gaussian Symplectic Ensemble (GSE) class
@dataclass(repr=False, eq=False, frozen=True, kw_only=True, slots=True)
class GSE(GaussianEnsemble):
    """Gaussian Symplectic Ensemble (GSE) class."""

    # Dyson index
    beta: int = field(init=False, default=4)

    def randm(self, offset: Optional[np.ndarray] = None) -> np.ndarray:
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
            real_rands = self._rng.standard_normal(bdim - i, dtype=self.real_dtype)

            # Generate a random array of standard normal values for imaginary parts
            imag_rands = self._rng.standard_normal(bdim - i, dtype=self.real_dtype)

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
            real_rands[1:] += self._rng.standard_normal(
                bdim - i - 1, dtype=self.real_dtype
            )

            # Generate random array of standard normal values for imaginary parts
            imag_rands = np.zeros(bdim - i, dtype=self.real_dtype)
            imag_rands[1:] += self._rng.standard_normal(
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
