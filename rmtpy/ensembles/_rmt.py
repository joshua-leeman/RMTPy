# rmtpy.ensembles._rmt.py


# =======================================
# 1. Imports
# =======================================
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterator, Optional

# Third-party imports
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.linalg import eigh, eigvalsh
from scipy.special import gamma


# =======================================
# 2. Random Matrix Theory Class
# =======================================
@dataclass(repr=False, eq=False, frozen=True, kw_only=True, slots=True)
class RMT(ABC):
    # Size of matrices
    dim: Optional[int] = field(init=False, default=None)

    # Data type of matrix elements
    dtype: np.dtype = field(default=np.dtype("complex64"))

    # Data type of real parts of matrix elements
    real_dtype: Optional[np.dtype] = field(init=False, default=None)

    # Amount of memory each matrix element takes in bytes
    matrix_memory: Optional[int] = field(init=False, default=None)

    # Amount of residual memory needed for each matrix in bytes
    resid_memory: int = field(init=False, default=0)

    # Random number generator seed
    seed: Optional[int] = field(default=None)

    # Random number generator
    _rng: Optional[np.random.Generator] = field(init=False, default=None)

    # Default ensemble argument names
    _ens_args: tuple[str] = field(init=False, default=("dim", "dtype"))

    @abstractmethod
    def randm(self, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate a random matrix of the ensemble."""
        raise NotImplementedError("randm method must be implemented in subclasses.")

    def __post_init__(self) -> None:
        """Finalize the initialization of the RMT class."""
        # Check if ensemble is valid
        self._check_ensemble()

        # Set data type for real parts
        if np.issubdtype(self.dtype, np.complexfloating):
            object.__setattr__(self, "real_dtype", np.dtype(self.dtype.char.lower()))
        else:
            object.__setattr__(self, "real_dtype", self.dtype)

        # Calculate memory size of matrix elements and store it in bytes
        object.__setattr__(self, "matrix_memory", self.dtype.itemsize * self.dim**2)

        # Initialize random number generator
        object.__setattr__(self, "_rng", np.random.default_rng(self.seed))

    @property
    def rng_state(self) -> dict[str, Any]:
        """State of the random number generator."""
        # Return state of random number generator
        return self._rng.bit_generator.state

    def _check_ensemble(self) -> None:
        """Check if the ensemble is valid."""
        # Check if dimension is set and valid
        if not isinstance(self.dim, int) or self.dim <= 0:
            raise ValueError("Dimension must set and be an integer.")

        # Normalize and check data type
        try:
            object.__setattr__(self, "dtype", np.dtype(self.dtype))
        except TypeError:
            raise TypeError("dtype must be a valid NumPy data type.")
        if not np.issubdtype(self.dtype, np.number):
            raise ValueError("dtype must be a number type.")

        # Check if seed is valid
        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError("Seed must be an integer if provided.")

    def _to_dict_str(self) -> str:
        """Return a dictionary of ensemble attributes"""
        # Begin dictionary with class name
        ens_dict = {"name": self.__class__.__name__}

        # Append all attributes to dictionary
        for arg in self._ens_args:
            # If argument is dtype, convert it to string
            if arg == "dtype":
                ens_dict[arg] = str(getattr(self, arg))
            # Otherwise, append the argument as is
            else:
                ens_dict[arg] = getattr(self, arg)

        # Return string representation of dictionary
        return repr(ens_dict)

    def _to_latex(self) -> str:
        """LaTeX representation of the ensemble."""
        # Return formatted LaTeX string
        return rf"$\textrm{{{self.__class__.__name__}}}\ D={self.dim}$"

    def _to_path(self) -> str:
        """Build path for simulation results."""
        # Begin output path with ensemble name
        ens_path = f"{self.__class__.__name__}"

        # Append remaining arguments to output directory
        for arg in self._ens_args:
            # If argument is float, convert it to string with 2 decimal places and replace dot with underscore
            if isinstance(getattr(self, arg), float):
                ens_path += f"/{arg}={getattr(self, arg):.2f}".replace(".", "_")
            # If argument is dtype, skip it
            elif arg == "dtype":
                continue
            # Otherwise, proceed with default string representation
            else:
                ens_path += f"/{arg}={getattr(self, arg)}"

        # Return formatted path
        return ens_path


# =======================================
# 3. Spectral Mixin
# =======================================
class SpectralMixin:
    def __init_subclass__(cls):
        """Check if subclasses define required attributes."""
        # Run parent class __init_subclass__ for compatibility
        super().__init_subclass__()

        # Check if required attributes are defined in class hierarchy
        for attr in ("E0", "beta", "degen", "dim", "dtype", "randm"):
            if not any(hasattr(base, attr) for base in cls.__mro__):
                raise TypeError(f"{cls.__name__} must define '{attr} to use mixin.")

    def eig_stream(self, realizs: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Iterator to stream eigensystem realizations."""
        # Allocate memory for random Hermitian matrix
        H = np.empty((self.dim, self.dim), dtype=self.dtype, order="F")

        # Loop over realizations
        for _ in range(realizs):
            # Generate random matrix
            self.randm(out=H)

            # Compute and yield eigenvalues and eigenvectors
            yield eigh(H, overwrite_a=True, check_finite=False)

    def eigvals_stream(self, realizs: int) -> Iterator[np.ndarray]:
        """Iterator to stream spectrum realizations."""
        # Allocate memory for random Hermitian matrix
        H = np.empty((self.dim, self.dim), dtype=self.dtype, order="F")

        # Loop over realizations
        for _ in range(realizs):
            # Generate random matrix
            self.randm(out=H)

            # Compute and yield eigenvalues
            yield eigvalsh(H, overwrite_a=True, check_finite=False)

    def pdf(self, eigval: np.ndarray) -> np.ndarray:
        """Average density of energy eigenstates."""

        # # Define function to compute PDF
        # @lru_cache(maxsize=1)
        # def numerical_pdf(realizs: int = 100, factor: float = 1.1) -> interp1d:
        #     """Create numerical PDF using eigenvalue realizations."""
        #     pass

        # # Return PDF values for given eigenvalues
        # return numerical_pdf()(eigval)

        # Raise NotImplementedError if not implemented
        raise NotImplementedError("Subclasses must implement this PDF method.")

    def cdf(self, eigval: np.ndarray) -> np.ndarray:
        """Average cumulative density of energy eigenstates."""

        # Define function to compute CDF
        @lru_cache(maxsize=1)
        def numerical_cdf(num_pts: int = 2**12, factor: int = 1.1) -> interp1d:
            """Create numerical CDF using trapezoidal rule."""
            # Generate grid of energies
            vals = factor * np.linspace(-self.E0, self.E0, num_pts)

            # Calculate PDF values
            pdf_vals = self.pdf(vals)

            # Compute CDF values using trapezoidal rule
            cdf_vals = cumulative_trapezoid(pdf_vals, vals, initial=0)

            # Create interpolation function
            cdf_interp = interp1d(vals, cdf_vals, bounds_error=False, fill_value=(0, 1))

            # Return interpolation function
            return cdf_interp

        # Return CDF values for given eigenvalues
        return numerical_cdf()(eigval)

    def unfold(self, eigval: np.ndarray) -> np.ndarray:
        """Unfold eigenvalues with the cumulative distribution function."""
        # Return unfolded eigenvalues
        return self.dim * (self.cdf(eigval) - self.cdf(np.array([0.0])))

    def univ_csff(self, tau: np.ndarray) -> np.ndarray:
        """Universal connected spectral form factor."""
        # Denote ensemble attributes
        dim = self.dim
        beta = self.beta
        degen = self.degen

        # Normalize unfolded times w.r.t. Heisenberg time 2π
        tau = tau / (2 * np.pi)

        # Return GOE connected spectral form factor if beta = 1
        if beta == 1:
            # Initialize csff array
            csff = np.empty_like(tau, dtype=self.real_dtype)

            # Handle case when tau is less than or equal to one
            m = tau <= 1
            csff[m] = tau[m] * (2 - np.log(2 * tau[m] + 1)) / dim

            # Handle case when tau is greater than one
            m = tau > 1
            csff[m] = (2 - tau[m] * np.log((2 * tau[m] + 1) / (2 * tau[m] - 1))) / dim

            # Return csff
            return csff

        # Return GUE connected spectral form factor if beta = 2
        elif beta == 2:
            return np.where(tau <= 1, tau / dim, 1 / dim)

        # Build GSE connected spectral form factor if beta = 4
        elif beta == 4:
            # Create default array for csff
            csff = np.full_like(tau, degen / dim)

            # Handle case when scaled tau is one
            csff[degen * tau == 1] = np.nan

            # Handle case when scaled tau is less than two
            m = (degen * tau < 2) & (degen * tau != 1)
            log_term = np.log(np.abs(degen * tau[m] - 1))
            csff[m] = degen * (tau[m] - tau[m] / 2 * log_term) / dim

            # Return GSE connected spectral form factor
            return csff

        # Return trivial csff for other Dyson indices
        else:
            return np.full_like(tau, 1 / dim)

    def wigner_surmise(self, s: np.ndarray) -> np.ndarray:
        """Wigner surmise for the nn-level spacing distribution."""
        # Denote ensemble attributes
        beta = self.beta
        degen = self.degen

        # If beta is 0, return Poisson distribution
        if beta == 0:
            return np.exp(-s)

        # Scale spacings by degeneracy
        s = s / degen

        # Calculate Wigner surmise
        a = gamma((beta + 2) / 2) ** (beta + 1) / gamma((beta + 1) / 2) ** (beta + 2)
        b = (gamma((beta + 2) / 2) / gamma((beta + 1) / 2)) ** 2

        # Return Wigner surmise at given spacings
        return 2 * a * s**beta * np.exp(-b * s**2) / degen


# =======================================
# 4. Chaotic Density Operator Mixin
# =======================================
class CDOMixin:
    def __init_subclass__(cls):
        """Check if subclasses define required attributes."""
        # Run parent class __init_subclass__ for compatibility
        super().__init_subclass__()

        # Check if required attributes are defined in class hierarchy
        for attr in ("dim", "dtype", "randm", "eig_stream", "unfold"):
            if not any(hasattr(base, attr) for base in cls.__mro__):
                raise TypeError(f"{cls.__name__} must define '{attr} to use mixin.")

    def evolve_states(
        self,
        state: np.ndarray,
        times: np.ndarray,
        realizs: int,
        unfold: bool = False,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # If out is None, allocate memory for output
        if out is None:
            out = np.empty((times.size, realizs, self.dim), dtype=self.dtype, order="F")

        # Loop over eigensystem realizations
        for r, (eigvals, eigvecs) in enumerate(self.eig_stream(realizs)):
            # Unfold eigenvalues if requested
            if unfold:
                eigvals = self.unfold(eigvals)

            # Rotate initial state into eigenbasis
            rotated_state = np.matmul(eigvecs.conj().T, state)

            # Construct time evolution operator in place
            np.outer(times, eigvals, out=out[:, r, :])
            np.multiply(-1j, out[:, r, :], out=out[:, r, :])
            np.exp(out[:, r, :], out=out[:, r, :])

            # Evolve state in eigenbasis in place
            np.multiply(out[:, r, :], rotated_state, out=out[:, r, :])

            # Rotate back to original basis
            np.matmul(out[:, r, :], eigvecs.T, out=out[:, r, :])

        # Return evolved states
        return out


# =======================================
# 5. Many Body Ensemble Class
# =======================================
@dataclass(repr=False, eq=False, frozen=True, kw_only=True, slots=True)
class ManyBodyEnsemble(RMT, SpectralMixin, CDOMixin):
    """Base class for quantum-chaotic many-body ensembles."""

    # Number of Majorana particles
    N: int

    # Interaction strength
    J: float = 1.0

    # Ground state energy
    E0: Optional[float] = field(init=False, default=None)

    # Dyson index
    beta: Optional[int] = field(init=False, default=None)

    # Universality class
    univ_class: Optional[str] = field(init=False, default=None)

    # Degeneracy of eigenvalues
    degen: Optional[int] = field(init=False, default=None)

    # Default ensemble argument names
    _ens_args: tuple[str, str] = field(init=False, default=("N", "J", "dtype"))

    def __post_init__(self) -> None:
        """Finalize the initialization of the ManyBodyEnsemble class."""
        # Check if Dyson index is set
        if self.beta is None:
            raise TypeError(f"{self.__class__.__name__} must define Dyson index.")

        # Calculate and set Hilbert space dimension
        object.__setattr__(self, "dim", 2 ** (self.N // 2 - 1))

        # Deterimine universality class
        if self.beta is not None:
            # Map Dyson index to universality class
            univ_classes = {0: "Poisson", 1: "GOE", 2: "GUE", 4: "GSE"}

            # Try to set universality class based on Dyson index
            try:
                object.__setattr__(self, "univ_class", univ_classes[self.beta])
            except KeyError:
                raise ValueError(f"Invalid Dyson index (beta): {self.beta}.") from None

        # Calculate and set ground state energy
        object.__setattr__(self, "E0", self.N * self.J)

        # Determine degeneracy of eigenvalues
        object.__setattr__(self, "degen", 2 if self.beta == 4 else 1)

        # Finish initialization of RMT instance
        super(ManyBodyEnsemble, self).__post_init__()

    def _check_ensemble(self) -> None:
        """Check if the ensemble is valid."""
        # Check number of Majorana fermions
        if not isinstance(self.N, int) or self.N <= 0 or self.N % 2 != 0:
            raise ValueError("N must be a positive even integer.")

        # Check interaction strength
        if not isinstance(self.J, (int, float)) or self.J <= 0:
            raise TypeError("J must be a positive integer or float.")

        # Call parent class _check_ensemble
        super(ManyBodyEnsemble, self)._check_ensemble()

    def _to_latex(self) -> str:
        """LaTeX representation of the ManyBodyEnsemble."""
        # Return formatted LaTeX string
        return rf"$\textrm{{{self.__class__.__name__}}}\ N={self.N}$"


# =======================================
# 6. Gaussian Ensemble Class
# =======================================
@dataclass(repr=False, eq=False, frozen=True, kw_only=True, slots=True)
class GaussianEnsemble(ManyBodyEnsemble):
    # Complex standard deviation of matrix elements
    sigma: Optional[float] = field(init=False, default=None)

    def __post_init__(self) -> None:
        """Finalize the initialization of the GaussianEnsemble class."""
        # Call parent class __post_init__
        super(GaussianEnsemble, self).__post_init__()

        # Calculate and set complex standard deviation
        object.__setattr__(self, "sigma", self.E0 / np.sqrt(2 * self.dim))

    def pdf(self, eigval: np.ndarray) -> np.ndarray:
        """Wigner semicircle probability density function."""
        # Normalize eigenvalues
        x = eigval / self.E0

        # Initialize PDF array
        pdf = np.zeros_like(x, dtype=self.real_dtype)

        # Create mask for eigenvalues
        mask = np.abs(x) < 1.0

        # Calculate PDF for eigenvalue array
        pdf[mask] = np.sqrt(1 - x[mask] * x[mask])
        pdf[mask] *= 2 / np.pi / self.E0

        # Return PDF array
        return pdf

    def cdf(self, eigval: np.ndarray) -> np.ndarray:
        """Cumulative distribution function of Wigner semicircle PDF."""
        # Normalize eigenvalues and clip to range [-1, 1]
        x = np.clip(eigval / self.E0, -1.0, 1.0)

        # Build CDF array
        cdf = np.sqrt(1 - x * x)
        cdf *= x
        cdf += np.arcsin(x)
        cdf /= np.pi
        cdf += 0.5

        # Return CDF array
        return cdf
