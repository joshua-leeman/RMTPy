# rmtpy/ensembles/gse.py

# Postponed evaluation of annotations
from __future__ import annotations

# Third-party imports
import numpy as np
from attrs import field, frozen
from numba import njit

# Local application imports
from ._base import GaussianEnsemble
from .gue import _create_gue_matrix


@njit(cache=True, fastmath=True)
def _add_gse_matrix(
    H: np.ndarray, d: int, rdtype: type, rng: np.random.Generator, std: float
) -> np.ndarray:
    """Add a GSE matrix to H with the given parameters."""

    # Alias block dimension
    b = d // 2

    # Loop over block diagonal indices
    for i in range(b):
        # Realize single complex normal with standard deviation
        diag = rng.standard_normal(None, rdtype) * std * 2

        # Add to ith diagonal elements of top-left block and bottom-right block
        H[i, i] += diag
        H[i + b, i + b] += diag

        # Realize d - i - 1 complex normals with standard deviation
        rands = (
            rng.standard_normal(d - 1 - i, rdtype)
            + 1j * rng.standard_normal(d - 1 - i, rdtype)
        ) * std

        # Add to upper triangular part of top-left block
        H[i, i + 1 : b] += rands

        # Add conjugate to lower triangular part of top-left block
        H[i + 1 : b, i] += np.conj(rands)

        # Add conjugate to upper triangular part of bottom-right block
        H[i + b, i + b + 1 :] += np.conj(rands)

        # Add to lower triangular part of bottom-right block
        H[i + b + 1 :, i + b] += rands

        # Again realize d - i - 1 complex normals with standard deviation
        rands[:] = (
            rng.standard_normal(d - 1 - i, rdtype)
            + 1j * rng.standard_normal(d - 1 - i, rdtype)
        ) * std

        # Add to upper triangular part of top-right block
        H[i, i + b + 1 :] += rands

        # Subtract from lower triangular part of top-right block
        H[i + 1 : b, i + b] -= rands

        # Subtract conjugate from upper triangular part of bottom-left block
        H[i + b, i + 1 : b] -= np.conj(rands)

        # Add conjugate to lower triangular part of bottom-left block
        H[i + b + 1 :, i] += np.conj(rands)


@njit(cache=True, fastmath=True)
def _create_skew_matrix(
    H: np.ndarray, b: int, rdtype: type, rng: np.random.Generator, std: float
) -> None:
    """Create a complex anti-symmetric matrix with the given parameters."""

    # Loop over diagonal indices
    for i in range(b):
        # Set ith diagonal element
        H[i, i] = 0.0

        # Set b - 1 - i standard normals for upper triangle
        H[i, i + 1 :] = (
            rng.standard_normal(b - 1 - i, rdtype)
            + 1j * rng.standard_normal(b - 1 - i, rdtype)
        ) * std

        # Set complex anti-symmetric elements in lower triangle
        H[i + 1 :, i] = -H[i, i + 1 :]


def _create_gse_matrix(
    H: np.ndarray, d: int, rdtype: type, rng: np.random.Generator, std: float
) -> np.ndarray:
    """Create a GSE matrix with the given parameters."""

    # Alias block dimension
    b = d // 2

    # Create GUE matrix in top-left block
    _create_gue_matrix(H[:b, :b], b, rdtype, rng, std)

    # Create conjugate of GUE matrix in bottom-right block
    np.conj(H[:b, :b], out=H[b:, b:])

    # Create complex anti-symmetric in top-right block
    _create_skew_matrix(H[:b, b:], b, rdtype, rng, std)

    # Set complex anti-symmetric elements in bottom-left block
    np.negative(H[:b, b:], out=H[b:, :b])
    np.conj(H[b:, :b], out=H[b:, :b])


# ----------------------------------
# Gaussian Symplectic Ensemble (GSE)
# ----------------------------------
@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class GSE(GaussianEnsemble):

    # Dyson index (for GSE is 4)
    beta: int = field(init=False, default=4, repr=False)

    def generate_matrix(
        self, out: np.ndarray | None = None, offset: np.ndarray | None = None
    ) -> np.ndarray:
        """Generate a random matrix from the GSE."""

        # Alias random number generator
        rng = self.rng

        # Alias data types of matrix elements
        cdtype = self.dtype.type
        rdtype = self.real_dtype.type

        # Alias dimension of matrix
        d = self.dim

        # Alias standard deviation of matrix elements
        std = self.sigma / 2

        # =============================================================

        # If offset is not None, add to provided matrix
        if offset is not None:
            # Alias provided matrix
            H = offset

            # Add GSE matrix to H
            _add_gse_matrix(H, d, rdtype, rng, std)

        # Otherwise, write to provided memory
        else:
            # Alias memory for output matrix
            if out is not None:
                # Alias provided matrix
                H = out
            else:
                # Create empty matrix
                H = np.empty((d, d), cdtype)

            # Create GSE matrix
            _create_gse_matrix(H, d, rdtype, rng, std)

        # Return GSE matrix
        return H
