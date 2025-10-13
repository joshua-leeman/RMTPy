# rmtpy/ensembles/poisson.py

# Postponed evaluation of annotations
from __future__ import annotations

# Standard library imports
from collections.abc import Callable, Iterator

# Third-party imports
import numpy as np
from attrs import field, frozen
from scipy.linalg.lapack import get_lapack_funcs

# Local imports
from .base.manybody import ManyBodyEnsemble


# ----------------
# Poisson Ensemble
# ----------------
@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class Poisson(ManyBodyEnsemble):
    # Standard deviation of eigenvalues
    sigma: float = field(init=False, repr=False)

    # Low-level LAPACK QR routines (for complex dtypes)
    __zgeqrf: Callable = field(init=False, repr=False)
    __zungqr: Callable = field(init=False, repr=False)

    @sigma.default
    def __sigma_default(self) -> float:
        """Default value for sigma."""
        # Calculate standard deviation based on E0
        return 2 * self.E0

    # Set LAPACK geqrf routine for QR decomposition
    @__zgeqrf.default
    def __zgeqrf_default(self) -> Callable:
        """Default low-level LAPACK QR routine for geqrf."""
        return get_lapack_funcs("geqrf", dtype=self.dtype)

    # Set LAPACK ungqr routine for generating Q from QR factorization
    @__zungqr.default
    def __zungqr_default(self) -> Callable:
        """Default low-level QR routine for ungqr."""
        return get_lapack_funcs("ungqr", dtype=self.dtype)

    @property
    def beta(self) -> int:
        """Dyson index of the Poisson ensemble."""
        return 0

    def generate(self, offset: np.ndarray | None = None) -> np.ndarray:
        """Generate a random matrix from the Poisson ensemble."""
        # If out is None, allocate memory for matrix
        if not isinstance(offset, np.ndarray):
            H = np.zeros((self.dim, self.dim), dtype=self.dtype, order="F")
        else:
            H: np.ndarray = offset

        # Allocate memory for complex Ginibre matrix
        M = np.empty((self.dim, self.dim), dtype=self.dtype, order="F")

        # Build complex Ginibre matrix
        M.real = self.rng.standard_normal((self.dim, self.dim), dtype=self.real_dtype)
        M.imag = self.rng.standard_normal((self.dim, self.dim), dtype=self.real_dtype)
        M /= np.sqrt(2)

        # In-place QR decomposition
        qr_fact, tau, _, _ = self.__zgeqrf(M, overwrite_a=True)
        U, _, _ = self.__zungqr(qr_fact, tau, overwrite_a=True)

        # Generate iid uniform random eigenvalues
        eigvals = self.rng.random(self.dim, dtype=self.real_dtype)
        eigvals -= 0.5
        eigvals *= self.sigma

        # Calculate Poisson ensemble matrix
        tmp = U * eigvals[None, :]
        M[:] = tmp @ U.conj().T
        H += M

        # Clip small real/imag parts to remove floating-point noise
        threshold = 1e-7
        H.real[np.abs(H.real) < threshold] = 0.0
        H.imag[np.abs(H.imag) < threshold] = 0.0

        # Return Poisson ensemble matrix
        return H

    def eig_stream(self, realizs: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Iterator to stream eigensystem realizations."""
        # Allocate memory for random eigenvalues and eigenvectors
        eigvals = np.empty(self.dim, dtype=self.real_dtype, order="F")
        U = np.empty((self.dim, self.dim), dtype=self.dtype, order="F")

        # Loop over realizations
        for _ in range(realizs):
            # Generate iid uniform random eigenvalues
            self.rng.random(self.dim, dtype=self.real_dtype, out=eigvals)
            eigvals -= 0.5
            eigvals *= self.sigma

            # Build complex Ginibre matrix
            U.real = self.rng.standard_normal(
                (self.dim, self.dim), dtype=self.real_dtype
            )
            U.imag = self.rng.standard_normal(
                (self.dim, self.dim), dtype=self.real_dtype
            )
            U /= np.sqrt(2)

            # Perform in-place QR decomposition
            qr_fact, tau, _, _ = self.__zgeqrf(U, overwrite_a=True)
            U, _, _ = self.__zungqr(qr_fact, tau, overwrite_a=True)

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
            self.rng.random(self.dim, dtype=self.real_dtype, out=eigvals)
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
