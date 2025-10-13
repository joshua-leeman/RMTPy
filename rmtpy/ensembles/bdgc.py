# rmtpy/ensembles/bdgc.py

# Postponed evaluation of annotations
from __future__ import annotations

# Third-party imports
import numpy as np
from attrs import field, frozen

# Local application imports
from ._base import GaussianEnsemble


# ----------------------------------------
# Bogoliubov-de Gennes C Ensemble (BdG(C))
# ----------------------------------------
@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class BdGC(GaussianEnsemble):

    # Dyson index (for BdG(C) is 2)
    beta: int = field(init=False, default=2, repr=False)

    @property
    def _dir_name(self) -> str:
        """Generate directory name used for storing BdG(C) instance data."""

        # Return formatted class name
        return "BdG_C"

    @property
    def _latex_name(self) -> str:
        """Generate LaTeX representation of BdG(C) class name."""

        # Return formatted LaTeX name
        return "\\textrm{{BdG(C)}}"

    def generate(self, offset: np.ndarray | None = None) -> np.ndarray:
        """Generate a random matrix from the BdG(C) ensemble."""

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

            # Add negative conjugate of top-left block to bottom-right block
            H[bdim + i, bdim + i :].real -= real_rands
            H[bdim + i, bdim + i :].imag += imag_rands
            H[bdim + i :, bdim + i].real -= real_rands
            H[bdim + i :, bdim + i].imag -= imag_rands

            # Generate random array of standard normal values for real parts
            real_rands = self.rng.standard_normal(bdim - i, dtype=self.real_dtype)

            # Generate random array of standard normal values for imaginary parts
            imag_rands = self.rng.standard_normal(bdim - i, dtype=self.real_dtype)

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
