# rmtpy.ensembles.syk.py


# =======================================
# 1. Imports
# =======================================
# Standard library imports
from __future__ import annotations
import itertools
from dataclasses import dataclass, field
from math import comb
from typing import Optional

# Third-party imports
import numpy as np
from scipy.sparse import csr_matrix

# Local application imports
from rmtpy.ensembles._rmt import ManyBodyEnsemble
from rmtpy.special import create_majorana_pairs


# =======================================
# 2. Ensemble
# =======================================
# Store class name for module
class_name = "SYK"


# Define Sachdev-Ye-Kitaev ensemble
@dataclass(repr=False, eq=False, frozen=True, kw_only=True, slots=True)
class SYK(ManyBodyEnsemble):
    """The Sachdev-Ye-Kitaev model class."""

    # SYK q-parameter
    q: int

    # Suppression factor
    eta: Optional[float] = field(init=False, default=None)

    # Standard deviation of couplings
    sigma: Optional[float] = field(init=False, default=None)

    # Majorana fermion operators
    majorana_pairs: Optional[tuple[tuple[Optional[csr_matrix], ...], ...]] = field(
        init=False, default=None
    )

    # Default ensemble argument names
    _ens_args: tuple[str, str] = field(init=False, default=("q", "N", "J", "dtype"))

    def __post_init__(self) -> None:
        # Create map to determine SYK Dyson index
        index = {(0, 0): 1, (0, 4): 4}.get((self.q % 4, self.N % 8), 2)

        # Set SYK ensemble's Dyson index
        object.__setattr__(self, "beta", index if self.q > 2 else 0)

        # Calculate suppression factor
        eta = np.sum(
            (-1) ** (self.q - k) * comb(self.q, k) * comb(self.N - self.q, self.q - k)
            for k in range(self.q + 1)
        ) / comb(self.N, self.q)

        # Set suppression factor
        object.__setattr__(self, "eta", eta)

        # Calculate standard deviation
        sigma = self.N * self.J * np.sqrt((1 - eta) / comb(self.N, self.q)) / 2

        # Set standard deviation
        object.__setattr__(self, "sigma", sigma)

        # Finish initialization of ManyBodyEnsemble instance
        super(SYK, self).__post_init__()

        # Residual memory is required to store the Majorana pairs in bytes
        object.__setattr__(self, "resid_memory", comb(self.N, 2) * (24 * self.dim + 6))

    def randm(self, offset: Optional[np.ndarray] = None) -> None:
        """Generate a random matrix from the SYK ensemble."""
        # If out is None, create a new zeroed array
        if offset is None:
            H = np.zeros((self.dim, self.dim), dtype=self.dtype, order="F")
        # If out is provided, zero it
        else:
            H = offset

        # Create Majorana operators if not already created
        if self.majorana_pairs is None:
            object.__setattr__(self, "majorana_pairs", create_majorana_pairs(N=self.N))

        # Pre-draw all random coefficients
        coeffs = self._rng.standard_normal(
            size=comb(self.N, self.q), dtype=self.real_dtype
        )

        # Retrieve indices for Hamiltonian terms
        indices = tuple(itertools.combinations(range(self.N), self.q))

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
            H[q_coo.row, q_coo.col] += coeff * q_coo.data

        # Scale Hamiltonian by standard deviation and global phase
        H *= 1j ** (self.q * (self.q - 1) / 2) * self.sigma

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

    def _check_ensemble(self) -> None:
        """Check if the SYK ensemble is valid."""
        # Check if instance is valid ManyBodyEnsemble instance
        super(SYK, self)._check_ensemble()

        # Check if SYK q-parameter is provided
        if self.q is None:
            raise ValueError("SYK q-parameter must be provided.")

        # Check if SYK parameters are valid
        if not isinstance(self.q, int) or self.q < 2 or self.q % 2 != 0:
            raise ValueError(f"SYK q-parameter must be an even integer.")

        elif self.N <= self.q:
            raise ValueError("Number of Majoranas must be greater than q-parameter.")

    def _to_latex(self) -> str:
        """LaTeX representation of the ManyBodyEnsemble."""
        # Return formatted LaTeX string
        return rf"$\textrm{{{self.__class__.__name__}}}_{self.q}\ N={self.N}$"
