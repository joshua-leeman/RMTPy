# rmtpy.ensembles.poisson.py
"""
This module contains the programs defining the Poisson ensemble.
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
from ._rmt import Ensemble


# =============================
# 2. Attributes
# =============================
# Class name for dynamic imports
class_name = "Poisson"

# Dyson index
beta = 0

# Degeneracy of eigenvalues
degen = 1


# =============================
# 3. Ensemble Class
# =============================
class Poisson(Ensemble):
    """
    The Poisson ensemble class.
    Inherits from the Ensemble class.

    Methods
    -------
    generate() -> np.ndarray
        Generate a random matrix from the Poisson ensemble.
    """

    def __init__(
        self,
        N: int = None,
        dim: int = None,
        J: float = 1.0,
        dtype: type = np.complex128,
    ) -> None:
        """
        Initialize the Poisson Ensemble.

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

        # Set Dyson index
        self._beta = beta

        # Initialize RMT ensemble
        super().__init__(N=N, dim=dim, J=J, dtype=dtype)

        # Calculate standard deviation of non-zero diagonal matrix elements
        self._sigma = self.N * self.J

    def generate(self) -> np.ndarray:
        """
        Return a random matrix from the Poisson ensemble.

        Returns
        -------
        np.ndarray
            Random matrix from the Poisson ensemble.
        """
        # Generate standard uniformly-distributed random eigenvalues
        eigvals = self._rng.random(self.dim, dtype=self.real_dtype)

        # Shift and scale eigenvalues by standard deviation in place
        eigvals -= 0.5
        eigvals *= self._sigma

        # Return diagonal matrix with eigenvalues
        return np.diag(eigvals).astype(self.dtype)

    def spectral_density(self, eigval: float) -> float:
        """
        Calculate the mean spectral density at a given eigenvalue.

        Parameters
        ----------
        eigval : float
            Eigenvalue at which to calculate the spectral density.

        Returns
        -------
        float
            Mean spectral density at the given eigenvalue.
        """
        # Return zero if eigenvalue is outside support
        if eigval < -0.5 * self.N * self.J or eigval > 0.5 * self.N * self.J:
            return 0.0

        # Calculate mean spectral density
        return 1.0 / self.N / self.J

    @property
    def sigma(self) -> float:
        """
        Standard deviation of the matrix elements.
        """
        return self._sigma
