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

# Third-party imports
import numpy as np

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
    ):
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

    def __repr__(self):
        """
        LaTeX representation of the SYK ensemble.
        """
        return rf"$\textrm{{{self.__class__.__name__}}}_{self.q}\ N={self.N}$"

    def __str__(self):
        """
        String representation of the SYK ensemble.
        """
        return f"{self.__class__.__name__} (q={self.q}, N={self.N}, scale={self.scale})"

    def _check_ensemble(self):
        pass

    @property
    def q(self):
        """
        Get the SYK parameter q.
        """
        return self._q
