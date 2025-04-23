# rmtpy.ensembles.syk.py
"""
This module contains the programs defining the Sachdev-Ye-Kitaev (SYK) ensemble.
It is grouped into the following sections:
    1. Imports
    2. Attributes
    3. Ensemble Class
"""


# =============================
# 1. Imports
# =============================
# Standard library imports
import itertools
from math import comb
from typing import List

# Third-party imports
import numpy as np
from scipy.sparse import csr_matrix, eye_array, kron

# Local application imports
from ._rmt import Ensemble


# =============================
# 2. Attributes
# =============================
# Class name for dynamic imports
class_name = "SYK"


# =============================
# 3. Ensemble Class
# =============================
class SYK(Ensemble):
    """
    The Sachdev-Ye-Kitaev (SYK) ensemble class.
    Inherits from the Ensemble class.

    Attributes
    ----------
    q : int
        Number of Majorana fermions.
    N : int
        Number of Majorana fermions.
    J : float
        Energy scale of interactions (default is 1.0).
    sigma : float
        Standard deviation of the couplings.

    Methods
    -------
    generate() -> np.ndarray
        Generate a random SYK Hamiltonian.
    spectral_density(eigval: float, num_terms: int = 1000) -> float
        Calculate the mean spectral density at a given eigenvalue.
    """

    def __init__(
        self,
        q: int = None,
        N: int = None,
        J: float = 1.0,
        dtype: type = np.complex128,
    ) -> None:
        """
        Intialize the SYK ensemble.

        Parameters
        ----------
        q : int
            Number of Majorana fermions.
        N : int
            Number of Majorana fermions.
        J : float, optional
            Energy scale of interactions (default is 1.0).
        dtype : type, optional
            Data type of the matrix elements (default is np.complex128).
        """
        # Set SYK q-parameter
        self._q = q

        # Determine Dyson index
        self._beta = {(0, 0): 1, (0, 4): 4}.get((q % 4, N % 8), 2) if q > 2 else 0

        # Set degeneracy of eigenvalues
        self._degen = 2 if self.beta == 4 else 1

        # Calculate suppression factor
        self._eta = np.sum(
            (-1) ** (q - k) * comb(q, k) * comb(N - q, q - k) for k in range(q + 1)
        ) / comb(N, q)

        # Calculate standard deviation of couplings
        self._sigma = N * J * np.sqrt((1 - self.eta) / comb(N, q)) / 2

        # Initialize ensemble class
        super().__init__(N=N, J=J, dtype=dtype)

        # Set order of SYK arguments
        self._arg_order = ["name", "q", "N", "J"]

    def __repr__(self) -> str:
        """
        String representation of the SYK ensemble.
        """
        return f"{self.__class__.__name__}(q={self.q}, N={self.N}, J={self.J})"

    def __str__(self) -> str:
        """
        LaTeX representation of the SYK ensemble.
        """
        return rf"$\textrm{{{self.__class__.__name__}}}_{self.q}\ N={self.N}$"

    def _check_ensemble(self) -> None:
        """
        Check if the SYK parameters are valid.

        Raises
        ------
        ValueError
            If the SYK parameters are invalid.
        """
        # Check if base ensemble parameters are valid
        super()._check_ensemble()

        # Checks if SYK q-parameter is provided
        if self.q is None:
            raise ValueError("SYK q-parameter must be provided.")

        # Check if SYK parameters are valid
        if self.q < 2 or self.q % 2 != 0:
            raise ValueError(
                f"Invalid SYK q-parameter: q={self.q}. Must be even and greater than or equal to 2."
            )
        elif self.N <= self.q:
            raise ValueError(f"Invalid N: N={self.N}. Must be greater than q={self.q}.")

    def _create_majoranas(self) -> List[csr_matrix]:
        """
        Create Majorana operators for the SYK ensemble.

        Returns
        -------
        List[csr_matrix]
            List of Majorana operators as sparse matrices.
        """
        # Create Pauli matrices
        pauli = [
            csr_matrix([[0, 1], [1, 0]], dtype=self.dtype),  # sigma_x
            csr_matrix([[0, -1j], [1j, 0]], dtype=self.dtype),  # sigma_y
            csr_matrix([[1, 0], [0, -1]], dtype=self.dtype),  # sigma_z
        ]

        # Create initial Majorana operators
        majorana_0 = pauli[:2]
        majorana_c0 = pauli[2]

        # With loop, build Majorana operators from the initial ones
        for i in range(self.N // 2 - 1):
            # Create identity matrix corresponding to old Majorana operators
            eye_mat = eye_array(2 ** (i + 1), format="csr", dtype=self.dtype)

            # Initialize new Majorana operators
            majorana = [None for _ in range(len(majorana_0) + 2)]

            # Create new Majorana operators
            for j in range(len(majorana_0)):
                majorana[j] = kron(pauli[0], majorana_0[j], format="csr")
            majorana[-2] = kron(pauli[0], majorana_c0, format="csr")
            majorana[-1] = kron(pauli[1], eye_mat, format="csr")

            # If not the last Majorana operators, update new as old
            # Else, return the last Majorana operators
            if i < self.N // 2 - 2:
                majorana_0 = majorana
                majorana_c0 = kron(pauli[2], eye_mat, format="csr")
            else:
                return majorana

    def generate(self) -> np.ndarray:
        """
        Return a random SYK Hamiltonian.

        Returns
        -------
        np.ndarray
            Random SYK Hamiltonian as a dense matrix.
        """
        # Create Majorana operators if not already created
        if not hasattr(self, "_majorana"):
            self._majorana = self._create_majoranas()

        # Pre-draw all random coefficients
        coeffs = self._rng.standard_normal(
            size=comb(self.N, self.q), dtype=self.real_dtype
        )

        # Initialize sparse SYK Hamiltonian
        H = csr_matrix((self.dim, self.dim), dtype=self.dtype)

        # Generate and sum q-body operators
        for coeff, idx_tuple in zip(
            coeffs, itertools.combinations(range(self.N), self.q)
        ):
            # Build q-body operator
            q_body = self.majorana[idx_tuple[0]]
            for idx in idx_tuple[1:]:
                q_body = q_body.dot(self.majorana[idx])

            # Add to Hamiltonian
            H += coeff * q_body[: self.dim, : self.dim]

        # Scale Hamiltonian by standard deviation and global phase
        H *= 1j ** (self.q * (self.q - 1) / 2) * self.sigma

        # Return SYK Hamiltonian as a dense matrix
        return H.toarray()

    def spectral_density(self, eigval: float, num_terms: int = 100) -> float:
        """
        Calculate the mean spectral density at a given eigenvalue.

        Parameters
        ----------
        eigval : float
            Eigenvalue at which to calculate the mean spectral density.

        Returns
        -------
        float
            Mean spectral density at the given eigenvalue.
        """
        # Return zero if eigenvalue is outside support
        if abs(eigval) > self._E0:
            return 0.0

        # Create vector of term indices
        k = np.arange(num_terms)

        # term1 = 1 - (2E/E0)**2 * eta**(k+1)/(1 + eta**(k+1))**2
        etak1 = self.eta ** (k + 1)
        term1 = 1 - (2 * eigval / self._E0) ** 2 * etak1 / (1.0 + etak1) ** 2

        # term2 = (1 - eta**(2*k+2))/(1 - eta**(2*k+1))
        term2 = (1.0 - self.eta ** (2 * k + 2)) / (1.0 - self.eta ** (2 * k + 1))

        # Sum logarithm of terms
        logP = np.log(term1) + np.log(term2)
        logP = np.sum(logP) + 0.5 * np.log(1.0 - self.eta)

        # Construct product and final result
        P = np.exp(logP) * np.sqrt(1.0 - (eigval / self._E0) ** 2)
        P /= np.pi * np.sqrt(comb(self.N, self.q)) * self.sigma

        # Return mean spectral density at eigenvalue
        return P

    @property
    def q(self):
        """
        Get the SYK parameter q.
        """
        return self._q

    @property
    def eta(self):
        """
        Get the suppression factor.
        """
        return self._eta

    @property
    def sigma(self):
        """
        Standard deviation of the couplings.
        """
        return self._sigma

    @property
    def majorana(self):
        """
        Majorana operators.
        """
        return self._majorana
