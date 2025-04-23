# rmtpy.ensembles.gue.py
"""
This module contains the programs defining the Gaussian Unitary Ensemble (GUE).
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
class_name = "GUE"

# Dyson index
beta = 2

# Degeneracy of eigenvalues
degen = 1


# =============================
# 3. Ensemble Class
# =============================
class GUE(Tenfold):
    """
    The Gaussian Unitary Ensemble (GUE) class.
    Inherits from the Tenfold class.

    Methods
    -------
    generate() -> np.ndarray
        Generate a random matrix from the GUE.
    """

    def __init__(
        self,
        N: int = None,
        dim: int = None,
        J: float = 1.0,
        dtype: type = np.complex128,
    ) -> None:
        """
        Initialize the Gaussian Unitary Ensemble (GUE).

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
        Return a random matrix from the GUE.

        Returns
        -------
        np.ndarray
            Random matrix from the GUE.
        """
        # Allocate memory for GUE matrix
        H = np.empty((self.dim, self.dim), dtype=self.dtype)

        # Generate standard normal numbers for real and imaginary parts
        H.real = self._rng.standard_normal(H.shape, dtype=self.real_dtype)
        H.imag = self._rng.standard_normal(H.shape, dtype=self.real_dtype)

        # Adjoint matrix in place
        np.add(H, H.T.conj(), out=H)

        # Halve and scale matrix by complex standard deviation in place
        H *= self.sigma / np.sqrt(2) / 2

        # Return GUE matrix
        return H
