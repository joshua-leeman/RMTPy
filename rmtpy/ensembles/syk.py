# rmtpy/ensembles/syk.py

# Postponed evaluation of annotations
from __future__ import annotations

# Standard library imports
from itertools import combinations
from math import comb, factorial

# Third-party imports
import numpy as np
from attrs import field, frozen
from attrs.validators import gt
from scipy.sparse import csr_matrix

# Local application imports
from ._base import ManyBodyEnsemble
from ..utils import create_majorana_pairs


# -----------------------------
# Sachdev-Ye-Kitaev Model (SYK)
# -----------------------------
@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class SYK(ManyBodyEnsemble):

    # SYK q-parameter
    q: int = field(converter=int, validator=gt(0))

    # Dyson index
    beta: int = field(init=False, repr=False)

    # Suppression factor
    eta: float = field(init=False, repr=False)

    # Standard deviation of couplings
    sigma: float = field(init=False, repr=False)

    # Ground state energy
    E0: float = field(init=False, repr=False)

    # Majorana fermion operators
    majorana_pairs: tuple[tuple[csr_matrix, ...], ...] = field(init=False, repr=False)

    # Validator to ensure q is even
    @q.validator
    def __q_validator(self, _, value: int) -> None:
        """Ensure that q is even."""

        # If q is not even, raise error
        if value % 2 != 0:
            raise ValueError(f"SYK q-parameter must be an even integer, got {value}")

    # Set Dyson index based on q and N
    @beta.default
    def __beta_default(self) -> int:
        """Default value for Dyson index."""

        # Alias number of Majorana fermions
        N = self.N

        # Alias q-parameter
        q = self.q

        # =================================================

        # Map to determine SYK Dyson index
        index_map = {(0, 0): 1, (0, 4): 4}  # (N, q) --> beta

        # Return appropriate index based on q and N
        return index_map.get((q % 4, N % 8), 2) if q > 2 else 0

    # Set suppression factor based on q and N
    @eta.default
    def __eta_default(self) -> float:
        """Default value for suppression factor."""

        # Alias number of Majorana fermions
        N = self.N

        # Alias q-parameter
        q = self.q

        # =================================================

        # First calculate product factor
        product = np.sum(
            (-1) ** (q - k) * comb(q, k) * comb(N - q, q - k) for k in range(q + 1)
        )

        # Calculate and return suppression factor
        return product / comb(N, q)

    # Set standard deviation based on N, J, eta, and q
    @sigma.default
    def __sigma_default(self) -> float:
        """Default value for standard deviation of couplings."""

        # Alias number of Majorana fermions
        N = self.N

        # Alias q-parameter
        q = self.q

        # Alias interaction strength
        J = self.J

        # =================================================

        # Calculate standard deviation based on N, J, eta, and q
        return J * np.sqrt(factorial(q - 1) / N ** (q - 1))

    # Construct 2-body Majorana operators
    @majorana_pairs.default
    def __majorana_pairs_default(self) -> tuple[tuple[csr_matrix, ...], ...]:
        """Default value for Majorana fermion operators."""

        # Alias number of Majorana fermions
        N = self.N

        # Alias Dyson index
        beta = self.beta

        # =================================================

        # Create and return tuple of 2-body Majorana operators
        return create_majorana_pairs(N=N, real_basis=(beta == 1))

    @E0.default
    def __E0_default(self) -> float:
        """Default value for ground state energy."""

        # Alias number of Majorana fermions
        N = self.N

        # Alias q-parameter
        q = self.q

        # Alias standard deviation of couplings
        sigma = self.sigma

        # Alias suppression factor
        eta = self.eta

        # =================================================

        # Return SYK ground state energy
        return 2 * sigma * np.sqrt(comb(N, q) / (1 - eta))

    @property
    def _dir_name(self) -> str:
        """Write name of SYK class with q parameter in directory format."""

        # Return class name as directory names
        return super()._dir_name + f"_{self.q}"

    @property
    def _latex_name(self) -> str:
        """Generate LaTeX representation of the SYK_q class name."""

        # Append q to the LaTeX name
        return super()._latex_name + f"_{self.q}"

    def generate_matrix(
        self, out: np.ndarray | None = None, offset: np.ndarray | None = None
    ) -> np.ndarray:
        """Generate a random matrix from the SYK ensemble."""

        # Alias random number generator
        rng = self.rng

        # Alias data types of matrix elements
        cdtype = self.dtype
        rdtype = self.real_dtype

        # Alias number of Majorana fermions
        N = self.N

        # Alias q-parameter
        q = self.q

        # Alias Dyson index
        beta = self.beta

        # Alias dimension of matrix
        d = self.dim

        # Alias standard deviation of couplings
        sigma = self.sigma

        # Alias tuple of Majorana pairs
        majorana_pairs = self.majorana_pairs

        # =================================================

        # If offset is not None, add to provided matrix
        if offset is not None:
            H = offset

        # If out is not None, fill with zeros
        elif out is not None:
            H = out
            H.fill(0.0)

        # Otherwise, create new zero matrix
        else:
            # If Dyson index is 1, allocate memory for real symmetric matrices
            if beta == 1:
                H = np.zeros((d, d), rdtype, order="F")

            # Else, allocate memory for complex Hermitian matrices
            else:
                H = np.zeros((d, d), cdtype, order="F")

        # Pre-draw all random coefficients
        coeffs = rng.standard_normal(comb(N, q), rdtype)

        # Scale coefficients by standard deviation
        coeffs *= sigma

        # Retrieve indices for Hamiltonian terms
        indices = tuple(combinations(range(N), q))

        # Generate and sum q-body operators
        for coeff, idx_tuple in zip(coeffs, indices):
            # Divide indices into pairs
            pairs = tuple((idx_tuple[i], idx_tuple[i + 1]) for i in range(0, q, 2))

            # Start q-body operator with first pair
            j0, k0 = pairs[0]
            q_body = majorana_pairs[j0][k0]

            # Multiply with remaining pairs
            for j, k in pairs[1:]:
                q_body = q_body.dot(majorana_pairs[j][k])

            # Store q-body operator as COO matrix
            if beta == 1:
                q_body_coo = q_body[:d, :d].real.tocoo()
            else:
                q_body_coo = q_body[:d, :d].tocoo()

            # Free memory of full q-body operator
            del q_body

            # Scale q-body operator by coefficient and phase factor
            if (q // 2) % 2 == 0:
                q_body_coo.data *= coeff
            else:
                q_body_coo.data *= 1j * coeff

            # Add q-body operator to Hamiltonian
            H[q_body_coo.row, q_body_coo.col] += q_body_coo.data

        # Return SYK Hamiltonian
        return H

    def pdf(self, eigval: np.ndarray, num_terms: int = 100) -> float:
        """SYK average spectral density."""

        # Alias data type of eigenvalues
        rdtype = self.real_dtype

        # Alias number of Majorana fermions
        N = self.N

        # Alias q-parameter
        q = self.q

        # Alias standard deviation of couplings
        sigma = self.sigma

        # Alias ground state energy
        E0 = self.E0

        # Alias suppression factor
        eta = self.eta

        # =================================================

        # Initialize probability distribution function (PDF) array
        pdf = np.zeros(eigval.shape, dtype=rdtype)

        # Create mask for non-trivial PDF values
        mask = np.abs(eigval) < E0

        # Create vector of term indices
        k = np.arange(num_terms)

        # term1 = 1 - (2E/E0)**2 * eta**(k+1)/(1 + eta**(k+1))**2
        etak1 = eta ** (k + 1)
        eigval_sq = (eigval[mask] ** 2)[:, None]
        term1 = 1 - (4 * eigval_sq / E0**2) * etak1 / (1.0 + etak1) ** 2

        # term2 = (1 - eta**(2*k+2))/(1 - eta**(2*k+1))
        term2 = (1.0 - eta ** (2 * k + 2)) / (1.0 - eta ** (2 * k + 1))

        # Sum logarithm of terms
        logP = np.log(term1) + np.log(term2)[None, :]
        logP = np.sum(logP, axis=1) + 0.5 * np.log(1.0 - eta)

        # Construct product and final result
        P = np.exp(logP) * np.sqrt(1.0 - (eigval[mask] / E0) ** 2)
        P /= np.pi * np.sqrt(comb(N, q)) * sigma

        # Store non-trivial PDF values
        pdf[mask] = P

        # Return PDF values
        return pdf
