# rmtpy/ensembles/goe.py

# Postponed evaluation of annotations
from __future__ import annotations

# Third-party imports
import numpy as np
from attrs import field, frozen
from numba import njit

# Local application imports
from ._base import GaussianEnsemble


@njit(cache=True, fastmath=True)
def _add_goe_matrix(
    H: np.ndarray, d: int, rdtype: type, rng: np.random.Generator, std: float
) -> np.ndarray:
    """Add a GOE matrix to H with the given parameters."""

    # Loop over diagonal indices
    for i in range(d):
        # Add ith diagonal element
        H[i, i] += rng.standard_normal(None, rdtype) * std * 2

        # Realize d - i - 1 normals with standard deviation
        rands = rng.standard_normal(d - i - 1, rdtype) * std

        # Add to upper triangular part
        H[i, i + 1 :] += rands

        # Add to lower triangular part
        H[i + 1 :, i] += rands


@njit(cache=True, fastmath=True)
def _create_goe_matrix(
    H: np.ndarray, d: int, rdtype: type, rng: np.random.Generator, std: float
) -> np.ndarray:
    """Create a GOE matrix with the given parameters."""

    # Loop over diagonal indices
    for i in range(d):
        # Set ith diagonal element
        H[i, i] = rng.standard_normal(None, rdtype) * std * 2

        # Set d - i - 1 standard normals to upper triangular part
        H[i, i + 1 :] = rng.standard_normal(d - i - 1, rdtype) * std

        # Set symmetric lower triangular part
        H[i + 1 :, i] = H[i, i + 1 :]


# ----------------------------------
# Gaussian Orthogonal Ensemble (GOE)
# ----------------------------------
@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class GOE(GaussianEnsemble):

    # Dyson index (for GOE is 1)
    beta: int = field(init=False, default=1, repr=False)

    def generate_matrix(
        self, out: np.ndarray | None = None, offset: np.ndarray | None = None
    ) -> np.ndarray:
        """Generate a random matrix from the GOE."""

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

            # Add GOE matrix to H
            _add_goe_matrix(H, d, rdtype, rng, std)

        # Otherwise, write to provided memory
        else:
            # Alias memory for output matrix
            if out is not None:
                # Alias provided matrix
                H = out
            else:
                # Create empty matrix
                H = np.empty((d, d), cdtype)

            # Create GOE matrix
            _create_goe_matrix(H, d, rdtype, rng, std)

        # Return GOE matrix
        return H
