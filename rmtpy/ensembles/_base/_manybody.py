# rmtpy/ensembles/base/manybody.py

# Postponed evaluation of annotations
from __future__ import annotations

# Standard library imports
from functools import lru_cache
from collections.abc import Iterator

# Third-party imports
import numpy as np
from attrs import field, frozen
from attrs.validators import gt
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.linalg import eigh, eigvalsh
from scipy.special import gamma

# Local application imports
from ._ensemble import Ensemble


# --------------------------------------------
# Random Many-Body Hamiltonian Generator Class
# --------------------------------------------
@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class ManyBodyEnsemble(Ensemble):

    # Dyson index (default is 0)
    beta: int | float = field(init=False, default=0, repr=False)

    # Number of Majorana particles
    N: int = field(
        converter=int, validator=gt(2), metadata={"dir_name": "N", "latex_name": "N"}
    )

    # Dimension of Hilbert space
    dim: int = field(init=False, repr=False)

    # Interaction strength
    J: float = field(
        default=1.0,
        converter=float,
        validator=gt(0),
        metadata={"dir_name": "J", "latex_name": "J"},
    )

    # Validator to ensure N is an even integer
    @N.validator
    def __N_validator(self, _, value: int) -> None:
        """Ensure N is an even integer."""

        if value % 2 != 0:
            raise ValueError(f"N must be an even integer, got {value}.")

    # Set dimension of Hilbert space based on number of Majorana particles
    @dim.default
    def __dim_default(self) -> int:
        """Calculate the dimension of the Hilbert space."""

        # Dimension of disconnected parity sector
        return 2 ** (self.N // 2 - 1)

    @property
    def E0(self) -> float:
        """Ground state energy of the ensemble."""

        # Return ground state energy based on N and J
        return self.N * self.J

    @property
    def univ_class(self) -> str | None:
        """Set the universality class based on the Dyson index."""

        # Map possible beta values to universality classes
        univ_map = {0.0: "Poisson", 1.0: "GOE", 2.0: "GUE", 4.0: "GSE"}

        # Return universality class if it exists
        return univ_map.get(float(self.beta), None)

    @property
    def degeneracy(self) -> int:
        """Determine the degeneracy of eigenvalues from the Dyson index."""

        # Return 2 if universality class is GSE, else return 1
        return 2 if self.univ_class == "GSE" else 1

    def eig_stream(self, realizs: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Iterator to stream eigensystem realizations."""

        # Allocate memory for random Hermitian matrices
        H = np.empty((self.dim, self.dim), dtype=self.dtype, order="F")

        # Loop over realizations
        for _ in range(realizs):
            # Zero out the matrix
            H.fill(0.0)

            # Generate random matrix
            self.generate(offset=H)

            # Compute and yield eigenvalues and eigenvectors
            yield eigh(H, overwrite_a=True, check_finite=False)

    def eigvals_stream(self, realizs: int) -> Iterator[np.ndarray]:
        """Iterator to stream spectrum realizations."""

        # Allocate memory for random Hermitian matrix
        H = np.empty((self.dim, self.dim), dtype=self.dtype, order="F")

        # Loop over realizations
        for _ in range(realizs):
            # Zero out the matrix
            H.fill(0.0)

            # Generate random matrix
            self.generate(offset=H)

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
        # raise NotImplementedError("Subclasses must implement this PDF method.")

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

    def wigner_surmise(self, s: np.ndarray) -> np.ndarray:
        """Wigner surmise for the nn-level spacing distribution."""

        # Denote ensemble attributes
        beta = self.beta
        degen = self.degeneracy

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

    def univ_csff(self, tau: np.ndarray) -> np.ndarray:
        """Universal connected spectral form factor."""

        # Denote ensemble attributes for convenience
        dim = self.dim
        beta = self.beta
        degen = self.degeneracy

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
