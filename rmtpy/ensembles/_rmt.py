# rmtpy.ensembles._rmt.py
"""
This module contains the classes for random matrix theory (RMT) ensembles.
It is grouped into the following sections:
    1. Imports
    2. RMT Class
    3. Spectral Mixin
    4. Ensemble Class
    5. Tenfold Class
"""


# =============================
# 1. Imports
# =============================
# Standard imports
from abc import ABC, abstractmethod
from typing import Tuple

# Third-party imports
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.linalg import eigvalsh
from scipy.special import gamma


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
        # Return unfolded eigenvalue about spectrum center mean
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
    def matrix_memory(self) -> int:
        """
        Memory size per matrix.
        """
        return self._matrix_memory

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


# =============================
# 3. Spectral Mixin
# =============================
class SpectralMixin:
    """
    Mixin class for spectral properties of RMT ensembles.

    Methods
    -------
    eigval_sample(realizs=1)
        Generate a sample of eigenvalues from the ensemble.
    nn_spacings(unfolded_eigvals=None)
        Calculate the nearest-neighbor level spacings of an eigenvalue sample.
    form_factors(times, unfolded_eigvals=None)
        Calculate the spectral form factors from an eigenvalue sample.
    wigner_surmise(s)
        Calculate the Wigner surmise for the given spacing.
    universal_csff(t)
        Calculate the universal connected spectral form factor at time t.
    """

    def eigval_sample(self, realizs: int = 1) -> np.ndarray:
        """
        Generate a sample of eigenvalues from the ensemble.

        Parameters
        ----------
        realizs : int, optional
            Number of realizations to sample (default is 1)

        Returns
        -------
        np.ndarray
            Sample of eigenvalues from the ensemble
        """
        # Allocate memory for realizations of eigenvalues
        eigenvalues = np.empty((realizs, self.dim), dtype=self.real_dtype)

        # Loop over realizations
        for r in range(realizs):
            # Compute eigenvalues of random matrix
            eigenvalues[r, :] = eigvalsh(
                self.generate(), overwrite_a=True, check_finite=False, driver="evr"
            )

        # Return eigenvalues
        return eigenvalues

    def nn_spacings(self, levels: np.ndarray = None) -> np.ndarray:
        """
        Calculate the nearest-neighbor level spacings of an eigenvalue sample.

        Parameters
        ----------
        unfolded_eigvals : np.ndarray, optional
            Unfolded eigenvalues (default is None)

        Returns
        -------
        np.ndarray
            Nearest-neighbor level spacings
        """
        # If ensemble dimension is one, raise error
        if self.dim == 1:
            raise ValueError(
                "Cannot compute nearest-neighbor spacings for 1D ensembles."
            )

        # If unfolded eigenvalues are not provided, generate them
        if levels is None:
            levels = np.vectorize(self.unfold)(self.eigval_sample())

        # Compute nearest-neighbor level spacings
        spacings = np.diff(levels, axis=1)

        # If degeneracy greater than one, clean spacings
        if self.degeneracy > 1:
            # Remove near-duplicate spacings
            spacings = spacings[:, 1 :: self.degeneracy]

            # Duplicate spacings with degeneracy
            spacings = np.repeat(spacings, self.degeneracy, axis=1)

        # Return nearest-neighbor level spacings
        return spacings

    def form_factors(
        self, times: np.ndarray, unfolded_eigvals: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the spectral form factors from an eigenvalue sample.

        Parameters
        ----------
        times : np.ndarray
            Array of time values for the form factors
        unfolded_eigvals : np.ndarray, optional
            Unfolded eigenvalues (default is None)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Spectral form factors and connected spectral form factors
        """
        # If unfolded eigenvalues are not provided, generate them
        if unfolded_eigvals is None:
            unfolded_eigvals = np.vectorize(self.unfold)(self.eigval_sample())

        # Calculate unfolded complex exponentials
        exponentials = np.multiply(times[:, None, None], unfolded_eigvals)
        exponentials = (-1j * 2 * np.pi) * exponentials
        np.exp(exponentials, out=exponentials)

        # Calculate partition function and its mean
        partition_func = np.sum(exponentials, axis=2)
        mean_partition_func = np.mean(partition_func, axis=1)

        # Calculate spectral form factor parts
        sff = np.abs(mean_partition_func) ** 2  # disconnected part
        csff = np.var(partition_func, axis=1)  # connected part
        sff += csff  # total

        # Normalize spectral form factors
        sff /= self.dim**2
        csff /= self.dim**2

        # Return spectral form factors
        return sff, csff

    def wigner_surmise(self, s: float) -> float:
        """
        Calculate the Wigner surmise for the given spacing.

        Parameters
        ----------
        s : float
            Spacing between unfolded eigenvalues

        Returns
        -------
        float
            Wigner surmise value for the given spacing
        """
        # If Dyson index is zero, return Poisson surmise
        if self.beta == 0:
            return np.exp(-s)

        # Calculate Wigner surmise
        a = (
            2
            * gamma((self.beta + 2) / 2) ** (self.beta + 1)
            / gamma((self.beta + 1) / 2) ** (self.beta + 2)
        )
        b = (gamma((self.beta + 2) / 2) / gamma((self.beta + 1) / 2)) ** 2

        # Return Wigner surmise at given spacing
        return a * s**self.beta * np.exp(-b * s**2)

    def universal_csff(self, t: float) -> float:
        """
        Calculate the universal connected spectral form factor at time t.

        Parameters
        ----------
        t : float
            Time parameter for the spectral form factor

        Returns
        -------
        float
            Universal connected spectral form factor at time t
        """
        # Return GOE connected spectral form factor if beta = 1
        if self.beta == 1:
            if t <= 1:
                return (2 * t - t * np.log(2 * t + 1)) / self.dim
            else:
                return (2 - t * np.log((2 * t + 1) / (2 * t - 1))) / self.dim

        # Return GUE connected spectral form factor if beta = 2
        elif self.beta == 2:
            if t <= 1:
                return t / self.dim
            else:
                return 1.0 / self.dim

        # Return GSE connected spectral form factor if beta = 4
        elif self.beta == 4:
            if t <= 2:
                return (t - t / 2 * np.log(abs(t - 1))) / self.dim
            else:
                return 2.0 / self.dim

        # Return unity for other Dyson indices
        else:
            return 1.0 / self.dim


# =============================
# 4. Ensemble Class
# =============================
class Ensemble(RMT, SpectralMixin):
    """
    Random matrix theory (RMT) ensemble class.
    Inherits from the RMT base class and the Spectral Mixin class.
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
        # Initialize RMT ensemble
        super().__init__(N=N, dim=dim, scale=scale, dtype=dtype)

        # Create cumulative density function
        self._create_cumulative_density()


# =============================
# 5. Tenfold Class
# =============================
class Tenfold(Ensemble):
    """
    Base class for tenfold-way RMT ensembles.

    Attributes
    ----------
    beta : int
        Dyson index (symmetry class)
    sigma : float
        Standard deviation of the matrix elements

    Methods
    -------
    spectral_density(eigenvalue)
        Calculate the mean spectral density at a given eigenvalue.
    """

    def __init__(
        self,
        beta: int,
        N: int = None,
        dim: int = None,
        scale: float = 1.0,
        dtype: type = np.complex128,
    ) -> None:
        """
        Initialize the tenfold ensemble.

        Parameters
        ----------
        beta : int
            Dyson index (symmetry class)
        N : int, optional
            Number of Majorana fermions (default is None)
        dim : int, optional
            Dimension of the matrix (default is None)
        scale : float, optional
            Energy scale (default is 1.0)
        dtype : type, optional
            Data type of the matrix (default is np.complex128)
        """
        # Initialize RMT ensemble
        super().__init__(N=N, dim=dim, scale=scale, dtype=np.float64)

        # Set Dyson index
        self._beta = beta

        # Calculate standard deviation of matrix elements
        self._sigma = self.scale / np.sqrt(2 * self.beta * self.dim)

    def spectral_density(self, eigenvalue: float) -> float:
        """
        Calculate the mean spectral density at eigenvalue.

        Parameters
        ----------
        eigenvalue : float
            Eigenvalue at which to evaluate the spectral density

        Returns
        -------
        float
            Spectral density at the given eigenvalue
        """
        # Calculate semi-circle spectral density
        if abs(eigenvalue) < self.scale:
            return np.sqrt(1 - (eigenvalue / self.scale) ** 2) / (
                np.pi * self.scale / 2
            )
        else:
            return 0.0

    @property
    def beta(self) -> int:
        """
        Dyson index (symmetry class).
            1: Orthogonal
            2: Unitary
            4: Symplectic
        """
        return self._beta

    @property
    def sigma(self) -> float:
        """
        Standard deviation of the matrix elements.
        """
        return self._sigma
