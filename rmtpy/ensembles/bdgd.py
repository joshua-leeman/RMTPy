# rmtpy/ensembles/bdgd.py

# Postponed evaluation of annotations
from __future__ import annotations

# Third-party imports
import numpy as np
from attrs import field, frozen
from numba import njit

# Local application imports
from ._base import GaussianEnsemble


@njit(cache=True, fastmath=True)
def _add_bdgd_matrix(
    H: np.ndarray, d: int, rdtype: type, rng: np.random.Generator, std: float
) -> np.ndarray:
    """Add a BdG(D) matrix to H with the given parameters."""

    # Loop over diagonal indices
    for i in range(d):
        # Realize d - i - 1 normals with standard deviation
        rands = rng.standard_normal(d - i - 1, rdtype) * std

        # Add to upper triangular part
        H[i, i + 1 :] += 1j * rands

        # Add conjugate to lower triangular part
        H[i + 1 :, i] -= 1j * rands


@njit(cache=True, fastmath=True)
def _create_bdgd_matrix(
    H: np.ndarray, d: int, rdtype: type, rng: np.random.Generator, std: float
) -> np.ndarray:
    """Create a BdG(D) matrix with the given parameters."""

    # Loop over diagonal indices
    for i in range(d):
        # Set ith diagonal element
        H[i, i] = 0.0

        # Set d - i - 1 standard normals
        H[i, i + 1 :] = 1j * rng.standard_normal(d - i - 1, rdtype) * std

        # Set conjugate off-diagonal elements
        H[i + 1 :, i] = np.conj(H[i, i + 1 :])


# ----------------------------------------
# Bogoliubov-de Gennes D Ensemble (BdG(D))
# ----------------------------------------
@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class BdGD(GaussianEnsemble):

    # Dyson index (for BdG(D) is 2)
    beta: int = field(init=False, default=2, repr=False)

    @property
    def _dir_name(self) -> str:
        """Generate directory name used for storing BdG(D) instance data."""

        # Return formatted class name
        return "BdG_D"

    @property
    def _latex_name(self) -> str:
        """Generate LaTeX representation of BdG(D) class name."""

        # Return formatted LaTeX name
        return "\\textrm{{BdG(D)}}"

    def generate_matrix(
        self, out: np.ndarray | None = None, offset: np.ndarray | None = None
    ) -> np.ndarray:
        """Generate a random matrix from the BdG(D) ensemble."""

        # Alias random number generator
        rng = self.rng

        # Alias data types of matrix elements
        cdtype = self.dtype.type
        rdtype = self.real_dtype.type

        # Alias dimension of matrix
        d = self.dim

        # Alias standard deviation of matrix elements
        std = self.sigma / np.sqrt(2)

        # =============================================================

        # If offset is not None, add to provided matrix
        if offset is not None:
            # Alias provided matrix
            H = offset

            # Add BdG(D) matrix to H
            _add_bdgd_matrix(H, d, rdtype, rng, std)

        # Otherwise, write to provided memory
        else:
            # Alias memory for output matrix
            if out is not None:
                # Alias provided matrix
                H = out
            else:
                # Create empty matrix
                H = np.empty((d, d), cdtype)

            # Create BdG(D) matrix
            _create_bdgd_matrix(H, d, rdtype, rng, std)

        # Return BdG(D) matrix
        return H
