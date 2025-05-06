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

    def randm(self, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate a random matrix from the BdGC ensemble."""
        # If out is None, allocate memory for matrix
        if out is None:
            H = np.empty((self.dim, self.dim), dtype=self.dtype, order="F")
        else:
            H = out

        # Compute block dimension
        bdim = self.dim // 2

        # Generate GUE in top-left block
        H[:bdim, :bdim].real = self._rng.standard_normal(
            (bdim, bdim), dtype=self.real_dtype
        )
        H[:bdim, :bdim].imag = self._rng.standard_normal(
            (bdim, bdim), dtype=self.real_dtype
        )
        np.add(
            H[:bdim, :bdim],
            H[:bdim, :bdim].T.conj(),
            out=H[:bdim, :bdim],
        )

        # Generate complex symmetric matrix in top-right block
        H[:bdim, bdim:].real = self._rng.standard_normal(
            (bdim, bdim), dtype=self.real_dtype
        )
        H[:bdim, bdim:].imag = self._rng.standard_normal(
            (bdim, bdim), dtype=self.real_dtype
        )
        np.add(
            H[:bdim, bdim:],
            H[:bdim, bdim:].T,
            out=H[:bdim, bdim:],
        )

        # Write bottom-left block as complex conjugate of top-right block
        np.conjugate(H[:bdim, bdim:], out=H[bdim:, :bdim])

        # Write bottom-right block as negative complex conjugate of top-left block
        np.conjugate(H[:bdim, :bdim], out=H[bdim:, bdim:])
        np.negative(H[bdim:, bdim:], out=H[bdim:, bdim:])

        # Halve and scale matrix by complex standard deviation in place
        H *= self.sigma / np.sqrt(2) / 2

        # Return BdG(C) matrix
        return H
