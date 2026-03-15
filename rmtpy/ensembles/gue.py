# rmtpy/ensembles/gue.py

# Postponed evaluation of annotations
from __future__ import annotations

# Third-party imports
import numpy as np
from attrs import field, frozen
from numba import njit

# Local application imports
from ._base import GaussianEnsemble


@njit(cache=True, fastmath=True)
def _add_gue_matrix(
    H: np.ndarray, d: int, rdtype: type, rng: np.random.Generator, std: float
) -> np.ndarray:
    """Add a GUE matrix to H with the given parameters."""

    # Loop over diagonal indices
    for i in range(d):
        # Add to ith diagonal element
        H[i, i] += rng.standard_normal(None, rdtype) * std * 2

        # Realize d - i - 1 complex normals with standard deviation
        rands = (
            rng.standard_normal(d - 1 - i, rdtype) * std
            + 1j * rng.standard_normal(d - 1 - i, rdtype) * std
        )

        # Add to upper triangular part
        H[i, i + 1 :] += rands

        # Add conjugate to lower triangular part
        H[i + 1 :, i] += np.conj(rands)


@njit(cache=True, fastmath=True)
def _create_gue_matrix(
    H: np.ndarray, d: int, rdtype: type, rng: np.random.Generator, std: float
) -> np.ndarray:
    """Create a GUE matrix with the given parameters."""

    # Loop over diagonal indices
    for i in range(d):
        # Set ith diagonal element
        H[i, i] = rng.standard_normal(None, rdtype) * std * 2

        # Set d - 1 - i standard normals to upper triangular part
        H[i, i + 1 :] = (
            rng.standard_normal(d - 1 - i, rdtype)
            + 1j * rng.standard_normal(d - 1 - i, rdtype)
        ) * std

        # Set conjugate lower triangular part
        H[i + 1 :, i] = np.conj(H[i, i + 1 :])


# -------------------------------
# Gaussian Unitary Ensemble (GUE)
# -------------------------------
@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class GUE(GaussianEnsemble):

    # Dyson index (for GUE is 2)
    beta: int = field(init=False, default=2, repr=False)

    def generate_matrix(
        self, out: np.ndarray | None = None, offset: np.ndarray | None = None
    ) -> np.ndarray:
        """Generate a random matrix from the GUE."""

        # Alias random number generator
        rng = self.rng

        # Alias data types of matrix elements
        cdtype = self.dtype.type
        rdtype = self.real_dtype.type

        # Alias dimension of matrix
        d = self.dim

        # Alias standard deviation of matrix elements
        std = self.sigma / 2

        # =================================================

        # If offset is not None, add to provided matrix
        if offset is not None:
            # Alias provided matrix
            H = offset

            # Add GUE matrix to H
            _add_gue_matrix(H, d, rdtype, rng, std)

        # Otherwise, write to provided memory
        else:
            # Alias memory for output matrix
            if out is not None:
                # Alias provided matrix
                H = out
            else:
                # Create empty matrix
                H = np.empty((d, d), cdtype)

            # Create GUE matrix
            _create_gue_matrix(H, d, rdtype, rng, std)

        # Return GUE matrix
        return H
