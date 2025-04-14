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
# 2. Ensemble Class
# =============================
class SYK(Ensemble):
    def __init__(
        self,
        q: int,
        N: int,
        scale: float = 1.0,
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
        scale : float, optional
            Energy scale (default is 1.0).
        dtype : type, optional
            Data type of the matrix elements (default is np.complex128).
        """
        # Set SYK-specific parameters
        self._q = q

        # Initialize ensemble class
        super().__init__(N=N, scale=scale, dtype=dtype)

        # Check if SYK parameters are valid
        self._check_ensemble()

        # Calculate suppression factor
        self._eta = np.sum(
            (-1) ** (self.q - k)
            * comb(self.q, k)
            * comb(self.N - k, self.q - k)
            / comb(self.N, self.q)
            for k in range(self.q + 1)
        )

        # Determine Dyson index
        self._beta = (
            {(0, 0): 1, (0, 4): 4}.get((self.q % 4, self.N % 8), 2) if q > 2 else 0
        )

        # Calculate standard deviation of matrix elements
        self._sigma = self.scale * np.sqrt((1 - self._eta) / comb(self.N, self.q)) / 2

        # Set order of SYK arguments
        self._arg_order = ["name", "q", "N", "scale"]

    def __repr__(self) -> str:
        """
        LaTeX representation of the SYK ensemble.
        """
        return rf"$\textrm{{{self.__class__.__name__}}}_{self.q}\ N={self.N}$"

    def __str__(self) -> str:
        """
        String representation of the SYK ensemble.
        """
        return f"{self.__class__.__name__} (q={self.q}, N={self.N}, scale={self.scale})"

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
            majorana = [None for _ in range(len(majorana_0) + 1)]

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
        Get the Dyson index.
        """
        return self._beta

    @property
    def sigma(self):
        """
        Get the standard deviation of the matrix elements.
        """
        return self._sigma
