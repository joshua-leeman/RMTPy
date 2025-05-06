# rmtpy.ensembles.poisson.py


# =======================================
# 1. Imports
# =======================================
# Standard library imports
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

# Third-party imports
import numpy as np
from scipy.linalg import eigh, eigvalsh
from scipy.linalg.lapack import get_lapack_funcs

# Local application imports
from rmtpy.ensembles._rmt import ManyBodyEnsemble


# =======================================
# 2. Ensemble
# =======================================
# Store class name for module
class_name = "Poisson"


# Define Poisson class
@dataclass(repr=False, eq=False, frozen=True, kw_only=True, slots=True)
class Poisson(ManyBodyEnsemble):
    """The Poisson class."""

    # Dyson index
    beta: int = field(init=False, default=0)

    # Complex standard deviation of matrix elements
    sigma: Optional[float] = field(init=False, default=None)

    # Low-level QR routines
    _zgeqrf: Optional[Callable] = None
    _zungqr: Optional[Callable] = None

    def __post_init__(self) -> None:
        """Finalize the initialization of the Poisson class."""
        # Call parent class __post_init__
        super(Poisson, self).__post_init__()

        # Set low-level QR routines
        object.__setattr__(self, "_zgeqrf", get_lapack_funcs("geqrf", dtype=self.dtype))
        object.__setattr__(self, "_zungqr", get_lapack_funcs("ungqr", dtype=self.dtype))

        # Calculate and set complex standard deviation
        object.__setattr__(self, "sigma", 2 * self.E0)

    def randm(self, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate a random matrix from the Poisson ensemble."""
        # If out is None, allocate memory for matrix
        if out is None:
            U = np.empty((self.dim, self.dim), dtype=self.dtype, order="F")
        else:
            U = out

        # Allocate memory for eigenvalues
        eigvals = np.empty(self.dim, dtype=self.real_dtype, order="F")

        # Build complex Ginibre matrix
        U.real = self._rng.standard_normal((self.dim, self.dim), dtype=self.real_dtype)
        U.imag = self._rng.standard_normal((self.dim, self.dim), dtype=self.real_dtype)
        U /= np.sqrt(2)

        # Perform in-place QR decomposition
        qr_fact, tau, _, _ = self._zgeqrf(U, overwrite_a=True)
        U, _, _ = self._zungqr(qr_fact, tau, overwrite_a=True)

        # Generate iid uniform random eigenvalues
        eigvals[:] = self._rng.random(self.dim, dtype=self.real_dtype)
        eigvals -= 0.5
        eigvals *= self.sigma

        # Calculate Poisson ensemble matrix
        np.dot(U, U.conj().T * eigvals[:, None], out=out)

        # Return Poisson ensemble matrix
        return out

    def eig_stream(self, realizs: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Iterator to stream eigensystem realizations."""
        # Allocate memory for random eigenvalues and eigenvectors
        eigvals = np.empty(self.dim, dtype=self.real_dtype, order="F")
        U = np.empty((self.dim, self.dim), dtype=self.dtype, order="F")

        # Loop over realizations
        for _ in range(realizs):
            # Generate iid uniform random eigenvalues
            self._rng.random(self.dim, dtype=self.real_dtype, out=eigvals)
            eigvals -= 0.5
            eigvals *= self.sigma

            # Build complex Ginibre matrix
            U.real = self._rng.standard_normal(
                (self.dim, self.dim), dtype=self.real_dtype
            )
            U.imag = self._rng.standard_normal(
                (self.dim, self.dim), dtype=self.real_dtype
            )
            U /= np.sqrt(2)

            # Perform in-place QR decomposition
            qr_fact, tau, _, _ = self._zgeqrf(U, overwrite_a=True)
            U, _, _ = self._zungqr(qr_fact, tau, overwrite_a=True)

            # Sort eigenvalues
            eigvals.sort()

            # Yield eigenvalues and eigenvectors
            yield eigvals, U

    def eigvals_stream(self, realizs: int) -> Iterator[np.ndarray]:
        """Iterator to stream spectrum realizations."""
        # Allocate memory for random eigenvalues
        eigvals = np.empty(self.dim, dtype=self.real_dtype, order="F")

        # Loop over realizations
        for _ in range(realizs):
            # Generate iid uniform random eigenvalues
            self._rng.random(self.dim, dtype=self.real_dtype, out=eigvals)
            eigvals -= 0.5
            eigvals *= self.sigma

            # Sort eigenvalues
            eigvals.sort()

            # Yield sorted eigenvalues
            yield eigvals

    def pdf(self, eigval: np.ndarray) -> np.ndarray:
        """Probability density function of the Poisson ensemble."""
        # Initialize distribution with zeros
        pdf = np.zeros_like(eigval, dtype=self.real_dtype)

        # Calculate non-zero elements
        pdf[np.abs(eigval) < self.E0] = 1 / 2 / self.E0

        # Return probability density function
        return pdf

    def cdf(self, eigval: np.ndarray) -> np.ndarray:
        """Cumulative distribution function of the Poisson ensemble."""
        # Initialize distribution with zeros
        cdf = np.zeros_like(eigval, dtype=self.real_dtype)

        # Calculate non-trivial elements
        mask = np.abs(eigval) < self.E0
        cdf[mask] = eigval[mask] / (2 * self.E0) + 0.5

        # Calculate remaining elements
        cdf[eigval > self.E0] = 1.0

        # Return cumulative distribution function
        return cdf
