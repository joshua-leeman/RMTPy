# rmtpy/ensembles/bdgc.py

# Postponed evaluation of annotations
from __future__ import annotations

# Third-party imports
import numpy as np
from attrs import field, frozen
from numba import njit

# Local application imports
from ._base import AZEnsemble
from .gue import _create_gue_matrix


@njit(cache=True, fastmath=True)
def _add_bdgc_matrix(
    H: np.ndarray, d: int, rdtype: type, rng: np.random.Generator, std: float
) -> np.ndarray:
    """Add a BdG(C) matrix to H with the given parameters."""

    # Alias block dimension
    b = d // 2

    # Loop over block diagonal indices
    for i in range(b):
        # Realize single complex normal with standard deviation
        diag = rng.standard_normal(None, rdtype) * std * 2

        # Add to ith diagonal elements of top-left block and bottom-right block
        H[i, i] += diag
        H[i + b, i + b] += diag

        # Realize b - i - 1 complex normals with standard deviation
        rands = (
            rng.standard_normal(b - 1 - i, rdtype)
            + 1j * rng.standard_normal(b - 1 - i, rdtype)
        ) * std

        # Add to upper triangular part of top-left block
        H[i, i + 1 : b] += rands

        # Add conjugate to lower triangular part of top-left block
        H[i + 1 : b, i] += np.conj(rands)

        # Subtract conjugate from upper triangular part of bottom-right block
        H[i + b, i + b + 1 :] -= np.conj(rands)

        # Subtract from lower triangular part of bottom-right block
        H[i + b + 1 :, i + b] -= rands

        # Again realize single complex normal with standard deviation
        diag = rng.standard_normal(None, rdtype) * std * 2

        # Add to ith diagonal elements of top-right block and bottom-left block
        H[i, i + b] += diag
        H[i + b, i] += diag

        # Again realize b - i - 1 complex normals with standard deviation
        rands[:] = (
            rng.standard_normal(b - 1 - i, rdtype)
            + 1j * rng.standard_normal(b - 1 - i, rdtype)
        ) * std

        # Add to upper triangular part of top-right block
        H[i, i + b + 1 :] += rands

        # Add from lower triangular part of top-right block
        H[i + 1 : b, i + b] += rands

        # Add conjugate from upper triangular part of bottom-left block
        H[i + b, i + 1 : b] += np.conj(rands)

        # Add conjugate to lower triangular part of bottom-left block
        H[i + b + 1 :, i] += np.conj(rands)


@njit(cache=True, fastmath=True)
def _create_symm_matrix(
    H: np.ndarray, b: int, rdtype: type, rng: np.random.Generator, std: float
) -> None:
    """Create a complex symmetric matrix with the given parameters."""

    # Loop over diagonal indices
    for i in range(b):
        # Set ith diagonal element
        H[i, i] = rng.standard_normal(None, rdtype) * std * 2

        # Set b - 1 - i standard normals for upper triangle
        H[i, i + 1 :] = (
            rng.standard_normal(b - 1 - i, rdtype)
            + 1j * rng.standard_normal(b - 1 - i, rdtype)
        ) * std

        # Set complex symmetric elements in lower triangle
        H[i + 1 :, i] = H[i, i + 1 :]


def _create_bdgc_matrix(
    H: np.ndarray, d: int, rdtype: type, rng: np.random.Generator, std: float
) -> np.ndarray:
    """Create a BdG(C) matrix with the given parameters."""

    # Alias block dimension
    b = d // 2

    # Create GUE matrix in top-left block
    _create_gue_matrix(H[:b, :b], b, rdtype, rng, std)

    # Create negative conjugate of GUE matrix in bottom-right block
    np.negative(H[:b, :b], out=H[b:, b:])
    np.conj(H[b:, b:], out=H[b:, b:])

    # Create complex symmetric in top-right block
    _create_symm_matrix(H[:b, b:], b, rdtype, rng, std)

    # Set conjugate of complex symmetric elements in bottom-left block
    np.conj(H[:b, b:], out=H[b:, :b])


# ----------------------------------------
# Bogoliubov-de Gennes C Ensemble (BdG(C))
# ----------------------------------------
@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class BdGC(AZEnsemble):

    # Dyson index (for BdG(C) is 2)
    beta: int = field(init=False, default=2, repr=False)

    @property
    def _dir_name(self) -> str:
        """Generate directory name used for storing BdG(C) instance data."""

        # Return formatted class name
        return "BdG_C"

    @property
    def _latex_name(self) -> str:
        """Generate LaTeX representation of BdG(C) class name."""

        # Return formatted LaTeX name
        return "\\textrm{{BdG(C)}}"

    def generate_matrix(
        self, out: np.ndarray | None = None, offset: np.ndarray | None = None
    ) -> np.ndarray:
        """Generate a random matrix from the BdG(C) ensemble."""

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

            # Add BdG(C) matrix to H
            _add_bdgc_matrix(H, d, rdtype, rng, std)

        # Otherwise, write to provided memory
        else:
            # Alias memory for output matrix
            if out is not None:
                # Alias provided matrix
                H = out
            else:
                # Create empty matrix
                H = np.empty((d, d), cdtype, order="F")

            # Create BdG(C) matrix
            _create_bdgc_matrix(H, d, rdtype, rng, std)

        # Return BdG(C) matrix
        return H
