# rmtpy.ensembles.bdgc.py


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
class_name = "BdGC"


# Define Bogoliubov-de Gennes C Ensemble (BdG(C)) class
@dataclass(repr=False, eq=False, frozen=True, kw_only=True, slots=True)
class BdGC(GaussianEnsemble):
    """Bogoliubov-de Gennes C Ensemble (BdG(C)) class."""

    # Dyson index
    beta: int = field(init=False, default=2)

    def _to_latex(self) -> str:
        """LaTeX representation of the ensemble."""
        # Return formatted LaTeX string
        return rf"$\textrm{{BdG(C)}}\ N={self.N}$"

    def randm(self, offset: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate a random matrix from the BdGC ensemble."""
        # If out is None, allocate memory for matrix
        if offset is None:
            H = np.zeros((self.dim, self.dim), dtype=self.dtype, order="F")
        else:
            H = offset

        # Compute block dimension
        bdim = self.dim // 2

        # Generate blocks of BdG(C) matrix
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

            # Add negative conjugate of top-left block to bottom-right block
            H[bdim + i, bdim + i :].real -= real_rands
            H[bdim + i, bdim + i :].imag += imag_rands
            H[bdim + i :, bdim + i].real -= real_rands
            H[bdim + i :, bdim + i].imag -= imag_rands

            # Generate random array of standard normal values for real parts
            real_rands = self._rng.standard_normal(bdim - i, dtype=self.real_dtype)

            # Generate random array of standard normal values for imaginary parts
            imag_rands = self._rng.standard_normal(bdim - i, dtype=self.real_dtype)

            # Scale random values by complex standard deviation
            real_rands *= self.sigma / 2
            imag_rands *= self.sigma / 2

            # Add real and imaginary parts of complex symmetric in top-right block
            H[i, bdim + i :].real += real_rands
            H[i, bdim + i :].imag += imag_rands
            H[i:bdim, bdim + i].real += real_rands
            H[i:bdim, bdim + i].imag += imag_rands

            # Add conjugate of top-right block to bottom-left block
            H[bdim + i, i:bdim].real += real_rands
            H[bdim + i, i:bdim].imag -= imag_rands
            H[bdim + i :, i].real += real_rands
            H[bdim + i :, i].imag -= imag_rands

        # Return BdG(C) matrix
        return H
