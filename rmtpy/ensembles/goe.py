# rmtpy.ensembles.goe.py
"""
This module contains the programs defining the Gaussian Orthogonal Ensemble (GOE).
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
class_name = "GOE"

# Dyson index
beta = 1

# Degeneracy of eigenvalues
degen = 1


# =============================
# 2. Ensemble Class
# =============================
class GOE(Tenfold):
    """
    The Gaussian Orthogonal Ensemble (GOE) class.
    Inherits from the Tenfold class.

    Methods
    -------
    generate() -> np.ndarray
        Generate a random matrix from the GOE.
    """

    def __init__(
        self,
        N: int = None,
        dim: int = None,
        J: float = 1.0,
        dtype: type = np.complex128,
    ) -> None:
        """
        Initialize the Gaussian Orthogonal Ensemble (GOE).

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
        Return a random matrix from the GOE.

        Returns
        -------
        np.ndarray
            Random matrix from the GOE.
        """
        # Allocate memory for GOE matrix
        H = np.empty((self.dim, self.dim), dtype=self.dtype)

        # Generate standard normal numbers for real parts and zeros for imaginary parts
        H.real = self._rng.standard_normal(H.shape, dtype=self.real_dtype)
        H.imag = np.zeros(H.shape, dtype=self.real_dtype)

        # Symmetrize matrix in-place
        np.add(H, H.T, out=H)

        # Halve and scale matrix by standard deviation in-place
        H *= self.sigma / 2

        # Return GOE matrix
        return H

    @property
    def degen(self) -> int:
        """
        Degeneracy of the ensemble's eigenvalues.
        """
        return self._degen
