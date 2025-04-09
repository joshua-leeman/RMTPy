# rmt.ensembles._rmt.py
"""
This module contains the classes for random matrix theory (RMT) ensembles.
It is grouped into the following sections:
    1. Imports
    2. RMT Class
"""


# =============================
# 1. Imports
# =============================
# Standard imports
from abc import ABC, abstractmethod

# Third-party imports
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d


# =============================
# 2. RMT Class
# =============================
class RMT(ABC):
    """
    Random matrix theory (RMT) ensemble base class.

    Attributes
    ----------
    N : int
        Number of Majorana fermions
    dim : int
        Dimension of the matrix
    scale : float
        Energy scale
    dtype : type
        Data type of the matrix
    real_dtype : type
        Real data type of the matrix
    beta : int
        Dyson index (symmetry class)
    degeneracy : int
        Degeneracy of the ensemble's eigenvalues

    Methods
    -------
    generate(out=None)
        Generate a random matrix instance of the ensemble.
    spectral_density(eigenvalue)
        Calculate the mean spectral density at a given eigenvalue.
    cumulative_density(eigenvalue)
        Calculate the cumulative density function value at a given eigenvalue.
    unfold(eigenvalue)
        Unfold an eigenvalue about the spectrum center.
    """

    def __init__(
        self,
        N: int = None,
        dim: int = None,
        scale: float = 1.0,
        dtype: type = np.complex128,
    ) -> None:
        """
        Initialize the RMT ensemble.

        Parameters
        ----------
        N : int, optional
            Number of Majorana fermions (default is None)
        dim : int, optional
            Dimension of the matrix (default is None)
        scale : float, optional
            Energy scale (default is 1.0)
        dtype : type, optional
            Data type of the matrix (default is np.complex128)
        """
        # Store ensemble parameters
        self._N = N
        self._dim = dim
        self._scale = scale

        # Store data type
        self._dtype = dtype

        # Check if ensemble is valid
        self._check_ensemble()

        # Store real data type
        self._real_dtype = self.dtype().real.dtype

        # Create random number generator
        self._rng = np.random.default_rng()

        # Store memory size per matrix
        self._matrix_memory = self.dim**2 * np.dtype(self.dtype).itemsize

        # Set default order of arguments
        self._arg_order = ["name", "N", "dim", "scale"]

    def __repr__(self) -> str:
        """
        LaTeX representation of the ensemble.
        """
        if self.N is None:
            return rf"$\textrm{{{self.__class__.__name__}}}\ D={self.dim}$"
        else:
            return f"$\textrm{{{self.__class__.__name__}}}\ N={self.N}$"

    def __str__(self) -> str:
        """
        String representation of the ensemble.
        """
        if self.N is None:
            return f"{self.__class__.__name__} (dim={self.dim}, scale={self.scale})"
        else:
            return f"{self.__class__.__name__} (N={self.N}, scale={self.scale})"

    def _check_ensemble(self) -> None:
        """
        Check if the ensemble parameters are valid.
        """
        # Check if dimension parameters are valid
        if self.N is not None:
            if self.N < 1 or self.N % 2 != 0:
                raise ValueError("Number of Majoranas must be a positive even integer.")
            if self.dim is not None and self.dim != 2 ** (self.N // 2 - 1):
                raise ValueError("N and dim must be consistent.")
        elif self.dim is not None:
            if self.dim < 1:
                raise ValueError("Dimension must be a positive integer.")
        else:
            raise ValueError("Either N or dim must be specified.")

        # If valid, clean N and dim inputs
        self._N = int(self.N) if self.N is not None else None
        self._dim = int(self.dim) if self.dim is not None else 2 ** (self.N // 2 - 1)

        # Check if energy scale is valid
        if not isinstance(self.scale, (int, float)) or self.scale <= 0:
            raise ValueError("Energy scale must be a positive number.")

        # Check if data type is valid
        try:
            np.dtype(self.dtype)
        except TypeError:
            raise TypeError("Data type must be a valid NumPy data type.")

    def _create_cumulative_density(
        self, num_pts: int = 2**16, multiplier: int = 3
    ) -> None:
        """
        Create numerical cumulative density function using trapezoidal integration.

        Parameters
        ----------
        num_pts : int, optional
            Number of points for the grid (default is 2**16)
        multiplier : int, optional
            Multiplier used to extend energy scale (default is 3)
        """
        # Create grid for cumulative trapezoidal integration
        eigen_grid = np.linspace(
            -multiplier * self.scale,
            multiplier * self.scale,
            num_pts,
        )

        # Calculate spectral density values
        density_values = self.dim * np.vectorize(self.spectral_density)(eigen_grid)

        # Compute numerical cumulative density function values
        cumulative_density_values = cumulative_trapezoid(
            density_values,
            eigen_grid,
            initial=0,
        )

        # Store cumulative density function
        self._cumulative_density = interp1d(eigen_grid, cumulative_density_values)

    @abstractmethod
    def generate(self, out: np.ndarray = None) -> np.ndarray:
        """
        Return an matrix instance of the ensemble.
        If `out` is provided, it will be filled with the generated matrix.

        Parameters
        ----------
        out : np.ndarray, optional
            Output matrix (default is None)

        Returns
        -------
        np.ndarray
            Random matrix instance of the ensemble
        """
        pass

    @abstractmethod
    def spectral_density(self, eigenvalue: float) -> float:
        """
        Return the ensemble's mean spectral density at eigenvalue.

        Parameters
        ----------
        eigenvalue : float
            Eigenvalue at which to evaluate the spectral density

        Returns
        -------
        float
            Spectral density at the given eigenvalue
        """
        pass

    def cumulative_density(self, eigenvalue: float) -> float:
        """
        Return the cumulative density function value at eigenvalue.

        Parameters
        ----------
        eigenvalue : float
            Eigenvalue at which to evaluate the cumulative density function

        Returns
        -------
        float
            Cumulative density function value at the given eigenvalue
        """
        # Return cumulative density function value
        return self._cumulative_density(eigenvalue)

    def unfold(self, eigenvalue: float) -> float:
        """
        Unfold eigenvalue about spectrum center.

        Parameters
        ----------
        eigenvalue : float
            Eigenvalue to unfold

        Returns
        -------
        float
            Unfolded eigenvalue
        """
        # Compute unfolded eigenvalue about the mean and return it
        return self.cumulative_density(eigenvalue) - self.dim // 2

    @property
    def N(self) -> int:
        """
        Number of Majorana fermions.
        """
        return self._N

    @property
    def dim(self) -> int:
        """
        Dimension of the matrix.
        """
        return self._dim

    @property
    def scale(self) -> float:
        """
        Energy scale.
        """
        return self._scale

    @property
    def dtype(self) -> type:
        """
        Data type of the matrix.
        """
        return self._dtype

    @property
    def real_dtype(self) -> type:
        """
        Real data type of the matrix.
        """
        return self._real_dtype

    @property
    @abstractmethod
    def beta(self) -> int:
        """
        Dyson index (symmetry class).
            1: Orthogonal
            2: Unitary
            4: Symplectic
        """
        pass

    @property
    @abstractmethod
    def degeneracy(self) -> int:
        """
        Degeneracy of the ensemble's eigenvalues.
        """
        pass
