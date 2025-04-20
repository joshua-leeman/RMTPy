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
from math import pi, comb, prod, sqrt
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
# 2. Ensemble Class
# =============================
class SYK(Ensemble):
    """
    The Sachdev-Ye-Kitaev (SYK) ensemble class.
    Inherits from the Ensemble class.

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
        # Set SYK parameters
        self._q = q
        self._N = N
        self._J = J

        # Calculate suppression factor
        self._eta = np.sum(
            (-1) ** (q - k) * comb(q, k) * comb(N - q, q - k) for k in range(q + 1)
        ) / comb(N, q)

        # Calculate standard deviation of matrix elements
        self._sigma = N * J * sqrt((1 - self.eta) / comb(N, q)) / 4

        # Initialize ensemble class
        super().__init__(N=N, J=J, dtype=dtype)

        # Check if SYK parameters are valid
        self._check_ensemble()

        # Determine Dyson index
        self._beta = (
            {(0, 0): 1, (0, 4): 4}.get((self.q % 4, self.N % 8), 2) if q > 2 else 0
        )

        # Set order of SYK arguments
        self._arg_order = ["name", "q", "N", "J"]

    def __repr__(self) -> str:
        """
        LaTeX representation of the SYK ensemble.
        """
        return rf"$\textrm{{{self.__class__.__name__}}}_{self.q}\ N={self.N}$"

    def __str__(self) -> str:
        """
        String representation of the SYK ensemble.
        """
        return f"{self.__class__.__name__} (q={self.q}, N={self.N}, J={self.J})"

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
        Calculate the mean spectral density at eigenvalue.

        Parameters
        ----------
        eigval : float
            Eigenvalue at which to calculate the mean spectral density.

        Returns
        -------
        float
            Mean spectral density at the given eigenvalue.
        """
        # Check if eigenvalue is within support
        if abs(eigval) < self._E0:
            # Approximate mean spectral density's infinite product
            product = prod(
                (
                    1
                    - (eigval / self._E0 * 2) ** 2
                    * self.eta ** (k + 1)
                    / (1 + self.eta ** (k + 1)) ** 2
                )
                * ((1 - self.eta ** (2 * k + 2)) / (1 - self.eta ** (2 * k + 1)))
                for k in range(num_terms)
            ) * sqrt(1 - self.eta)

            # Return SYK mean spectral density at eigenvalue
            return (
                product
                * sqrt(1 - (eigval / self._E0) ** 2)
                / (pi * sqrt(comb(self.N, self.q)) * self.sigma)
            )
        else:
            # If eigenvalue is outside support, return 0
            return 0.0

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
    def beta(self):
        """
        Dyson index (symmetry class).
            1: Orthogonal
            2: Unitary
            4: Symplectic
        """
        return self._beta

    @property
    def sigma(self):
        """
        Standard deviation of the matrix elements.
        """
        return self._sigma

    @property
    def majorana(self):
        """
        Majorana operators.
        """
        return self._majorana

    @property
    def degeneracy(self):
        """
        Degeneracy of the ensemble's eigenvalues.
        """
        return 2 if self.beta == 4 else 1
