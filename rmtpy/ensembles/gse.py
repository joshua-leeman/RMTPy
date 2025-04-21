# rmtpy.ensembles.gse.py
"""
This module contains the programs defining the Gaussian Symplectic Ensemble (GSE).
It is grouped into the following sections:
    1. Imports
    2. Attributes
    3. Ensemble Class
"""

# =============================
# 1. Imports
# =============================
# Third-party imports
import numpy as np

# Local application imports
from ._rmt import Tenfold


# =============================
# 2. Attributes
# =============================
# Class name for dynamic imports
class_name = "GSE"

# Dyson index
beta = 4

# Degeneracy of eigenvalues
degen = 2


# =============================
# 3. Ensemble Class
# =============================
class GSE(Tenfold):
    """
    The Gaussian Symplectic Ensemble (GSE) class.
    Inherits from the Tenfold class.

    Attributes
    ----------
    sigma : float
        Standard deviation of the matrix elements

    Methods
    -------
    generate() -> np.ndarray
        Generate a random matrix from the GSE.
    """

    def __init__(
        self,
        N: int = None,
        dim: int = None,
        J: float = 1.0,
        dtype: type = np.complex128,
    ) -> None:
        """
        Initialize the Gaussian Symplectic Ensemble (GSE).

        Parameters
        ----------
        N : int, optional
            Number of Majorana fermions
        dim : int, optional
            Dimension of the matrix
        J : float, optional
            Energy scale of interactions (default is 1.0)
        dtype : type, optional
            Data type of the matrix elements (default is np.complex128)
        """
        # Set degeneracy of eigenvalues
        self._degen = degen

        # Initialize tenfold ensemble
        super().__init__(beta=beta, N=N, dim=dim, J=J, dtype=dtype)

        # Calculate standard deviation of matrix element parts
        self._sigma = self.N * self.J / 2 / np.sqrt(4 * self.dim)

    def generate(self) -> np.ndarray:
        """
        Return a random matrix from the GSE.

        Returns
        -------
        np.ndarray
            Random matrix from the GSE.
        """
        # Compute block dimension
        bdim = self.dim // 2

        # Allocate memory for GSE matrix
        H = np.empty((self.dim, self.dim), dtype=self.dtype)

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

        # Generate complex anti-symmetric matrix in top-right block
        H[:bdim, bdim:].real = self._rng.standard_normal(
            (bdim, bdim), dtype=self.real_dtype
        )
        H[:bdim, bdim:].imag = self._rng.standard_normal(
            (bdim, bdim), dtype=self.real_dtype
        )
        np.subtract(
            H[:bdim, bdim:],
            H[:bdim, bdim:].T,
            out=H[:bdim, bdim:],
        )

        # Write bottom-left block as negative complex conjugate of top-right block
        np.conjugate(H[:bdim, bdim:], out=H[bdim:, :bdim])
        np.negative(H[bdim:, :bdim], out=H[bdim:, :bdim])

        # Write bottom-right block as complex conjugate of top-left block
        np.conjugate(H[:bdim, :bdim], out=H[bdim:, bdim:])

        # Halve and scale matrix by standard deviation in place
        H *= self.sigma / 2

        # Return GSE matrix
        return H

    @property
    def sigma(self) -> float:
        """
        Standard deviation of the matrix elements.
        """
        return self._sigma
