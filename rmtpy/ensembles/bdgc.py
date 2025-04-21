# rmtpy.ensembles.bdgc.py
"""
This module contains the programs defining the Bogoliubov-de Gennes C Ensemble (BdG(C)).
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
class_name = "BdGC"

# Dyson index
beta = 2

# Degeneracy of eigenvalues
degen = 1


# =============================
# 3. Ensemble Class
# =============================
class BdGC(Tenfold):
    """
    The Bogoliubov-de Gennes C Ensemble (BdG(C)) class.
    Inherits from the Tenfold class.

    Attributes
    ----------
    sigma : float
        Standard deviation of the matrix elements

    Methods
    -------
    generate() -> np.ndarray
        Generate a random matrix from the BdG(C).
    """

    def __init__(
        self,
        N: int = None,
        dim: int = None,
        J: float = 1.0,
        dtype: type = np.complex128,
    ) -> None:
        """
        Initialize the Bogoliubov-de Gennes C Ensemble (BdG(C)).

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

        # Set standard deviation of matrix element parts
        self._sigma = self.N * self.J / 2 / np.sqrt(4 * self.dim)

    def generate(self) -> np.ndarray:
        """
        Return a random matrix from the BdG(C).

        Returns
        -------
        np.ndarray
            Random matrix from the BdG(C).
        """
        # Compute block dimension
        bdim = self.dim // 2

        # Allocate memory for BdG(C) matrix
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

        # Halve and scale matrix by standard deviation in place
        H *= self.sigma / 2

        # Return BdG(C) matrix
        return H

    @property
    def sigma(self) -> float:
        """
        Standard deviation of the matrix elements.
        """
        return self._sigma
