# rmtpy/ensembles/syk.py

# Postponed evaluation of annotations
from __future__ import annotations

# Standard library imports
from itertools import combinations
from math import comb

# Third-party imports
import numpy as np
from attrs import field, frozen
from attrs.validators import gt
from scipy.sparse import csr_matrix

# Local application imports
from ._base import ManyBodyEnsemble
from .utils import create_majorana_pairs


# -----------------------------
# Sachdev-Ye-Kitaev Model (SYK)
# -----------------------------
@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class SYK(ManyBodyEnsemble):

    # SYK q-parameter
    q: int = field(converter=int, validator=gt(0))

    # Suppression factor
    eta: float = field(init=False, repr=False)

    # Standard deviation of couplings
    sigma: float = field(init=False, repr=False)

    # Majorana fermion operators
    majorana_pairs: tuple[tuple[csr_matrix, ...], ...] | None = field(
        init=False, repr=False, default=None
    )

    # Validator to ensure q is even
    @q.validator
    def __q_validator(self, _, value: int) -> None:
        """Ensure that q is even."""

        # If q is not even, raise error
        if value % 2 != 0:
            raise ValueError(f"SYK q-parameter must be an even integer, got {value}")

    # Set suppression factor based on q and N
    @eta.default
    def __eta_default(self) -> float:
        """Default value for suppression factor."""

        # First calculate product factor
        product = np.sum(
            (-1) ** (self.q - k) * comb(self.q, k) * comb(self.N - self.q, self.q - k)
            for k in range(self.q + 1)
        )

        # Calculate and return suppression factor
        return product / comb(self.N, self.q)

    # Set standard deviation based on N, J, eta, and q
    @sigma.default
    def __sigma_default(self) -> float:
        """Default value for standard deviation of couplings."""

        # Calculate standard deviation based on N, J, eta, and q
        return self.N * self.J * np.sqrt((1 - self.eta) / comb(self.N, self.q)) / 2

    @property
    def beta(self) -> int:
        """Dyson index of the SYK model."""

        # Map to determine SYK Dyson index
        index_map = {(0, 0): 1, (0, 4): 4}  # (N, q) --> beta

        # Return appropriate index based on q and N
        return index_map.get((self.q % 4, self.N % 8), 2) if self.q > 2 else 0

    @property
    def _dir_name(self) -> str:
        """Write name of ensemble class in directory format."""

        # Return class name as directory names
        return super()._dir_name + f"_{self.q}"

    @property
    def _latex_name(self) -> str:
        """Generate LaTeX representation of the ensemble class name."""

        # Append q to the LaTeX name
        return super()._latex_name + f"_{self.q}"

    def generate(self, offset: np.ndarray | None = None) -> np.ndarray:
        """Generate a random matrix from the SYK ensemble."""

        # If offset is None, create a new zeroed array
        if offset is None:
            H = np.zeros((self.dim, self.dim), dtype=self.dtype, order="F")
        else:
            H = offset

        # If Majorana pairs not set, create them
        if self.majorana_pairs is None:
            object.__setattr__(self, "majorana_pairs", create_majorana_pairs(N=self.N))

        # Pre-draw all random coefficients
        coeffs = self.rng.standard_normal(
            size=comb(self.N, self.q), dtype=self.real_dtype
        )

        # Scale coefficients by standard deviation
        coeffs *= self.sigma

        # Retrieve indices for Hamiltonian terms
        indices = tuple(combinations(range(self.N), self.q))

        # Generate and sum q-body operators
        for coeff, idx_tuple in zip(coeffs, indices):
            # Divide indices into pairs
            pairs = tuple((idx_tuple[i], idx_tuple[i + 1]) for i in range(0, self.q, 2))

            # Start q-body operator with first pair
            j0, k0 = pairs[0]
            q_body = self.majorana_pairs[j0][k0]

            # Multiply with remaining pairs
            for j, k in pairs[1:]:
                q_body = q_body.dot(self.majorana_pairs[j][k])

            # Store q-body operator as COO matrix
            q_coo = q_body[: self.dim, : self.dim].tocoo()

            # Add q-body operator to Hamiltonian
            H[q_coo.row, q_coo.col] += (
                1j ** (self.q * (self.q - 1) / 2) * coeff * q_coo.data
            )

        # Return SYK Hamiltonian
        return H

    def pdf(self, eigval: np.ndarray, num_terms: int = 100) -> float:
        """SYK average spectral density."""

        # Initialize probability distribution function (PDF) array
        pdf = np.zeros(eigval.shape, dtype=self.real_dtype)

        # Create mask for non-trivial PDF values
        mask = np.abs(eigval) < self.E0

        # Create vector of term indices
        k = np.arange(num_terms)

        # term1 = 1 - (2E/E0)**2 * eta**(k+1)/(1 + eta**(k+1))**2
        etak1 = self.eta ** (k + 1)
        eigval_sq = (eigval[mask] ** 2)[:, None]
        term1 = 1 - (4 * eigval_sq / self.E0**2) * etak1 / (1.0 + etak1) ** 2

        # term2 = (1 - eta**(2*k+2))/(1 - eta**(2*k+1))
        term2 = (1.0 - self.eta ** (2 * k + 2)) / (1.0 - self.eta ** (2 * k + 1))

        # Sum logarithm of terms
        logP = np.log(term1) + np.log(term2)[None, :]
        logP = np.sum(logP, axis=1) + 0.5 * np.log(1.0 - self.eta)

        # Construct product and final result
        P = np.exp(logP) * np.sqrt(1.0 - (eigval[mask] / self.E0) ** 2)
        P /= np.pi * np.sqrt(comb(self.N, self.q)) * self.sigma

        # Store non-trivial PDF values
        pdf[mask] = P

        # Return PDF values
        return pdf
