# rmtpy.ensembles.bdgd.py
"""
This module contains the programs defining the Bogoliubov-de Gennes D Ensemble (BdG(D)).
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
class_name = "BdGD"

# Dyson index
beta = 2

# Degeneracy of eigenvalues
degen = 1


# =============================
# 3. Ensemble Class
# =============================
class BdGD(Tenfold):
    """
    The Bogoliubov-de Gennes D Ensemble (BdG(D)) class.
    Inherits from the Tenfold class.

    Methods
    -------
    generate() -> np.ndarray
        Generate a random matrix from the BdG(D).
    """

    def __init__(
        self,
        N: int = None,
        dim: int = None,
        J: float = 1.0,
        dtype: type = np.complex128,
    ) -> None:
        """
        Initialize the Bogoliubov-de Gennes D Ensemble (BdG(D)).

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

    def generate(self) -> np.ndarray:
        """
        Return a random matrix from the BdG(D).

        Returns
        -------
        np.ndarray
            Random matrix from the BdG(D).
        """
        # Allocate memory for BdG(D) matrix
        H = np.empty((self.dim, self.dim), dtype=self.dtype)

        # Generate standard normal numbers for imaginary parts and zeros for real parts
        H.real = np.zeros(H.shape, dtype=self.real_dtype)
        H.imag = self._rng.standard_normal(H.shape, dtype=self.real_dtype)

        # Anti-symmetrize matrix in place
        np.subtract(H, H.T, out=H)

        # Halve and scale matrix by real standard deviation in place
        H *= self.sigma / 2

        # Return BdG(D) matrix
        return H
