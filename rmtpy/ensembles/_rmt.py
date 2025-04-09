# rmt.ensembles._rmt.py


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
    def __init__(
        self,
        N: int = None,
        dim: int = None,
        scale: float = 1.0,
        dtype: type = np.complex128,
    ):
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

    def __repr__(self) -> str:
        if self.N is None:
            return rf"$\textrm{{{self.__class__.__name__}}}\ D={self.dim}$"
        else:
            return f"$\textrm{{{self.__class__.__name__}}}\ N={self.N}$"

    def __str__(self) -> str:
        if self.N is None:
            return f"{self.__class__.__name__} (dim={self.dim}, scale={self.scale})"
        else:
            return f"{self.__class__.__name__} (N={self.N}, scale={self.scale})"

    def _check_ensemble(self):
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

    @abstractmethod
    def generate(self, out=None) -> np.ndarray:
        pass

    @abstractmethod
    def spectral_density(self, eigenvalue: float) -> float:
        pass

    def _prepare_cumulative_density(self, num_pts: int = 2**16, multiplier: int = 3):
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
        self.cumulative_density = interp1d(eigen_grid, cumulative_density_values)

    @property
    def N(self) -> int:
        return self._N

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def scale(self) -> float:
        return self._scale

    @property
    def dtype(self) -> type:
        return self._dtype

    @property
    def real_dtype(self) -> type:
        return self._real_dtype

    @property
    @abstractmethod
    def beta(self) -> int:
        pass

    @property
    @abstractmethod
    def degeneracy(self) -> int:
        pass
