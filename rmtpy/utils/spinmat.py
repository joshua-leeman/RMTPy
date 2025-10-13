# rmtpy/utils/spinmat.py

# Third-party imports
import numpy as np
from scipy.sparse import csr_matrix, eye_array, kron


# ---------------------------
# Construct Majorana Fermions
# ---------------------------
def create_majoranas(N: int) -> tuple[csr_matrix, ...]:
    """Create list of N Majorana fermion operators."""
    # Create Pauli matrices
    pauli = (
        csr_matrix([[0, 1], [1, 0]], dtype=np.complex64),  # sigma_x
        csr_matrix([[0, -1j], [1j, 0]], dtype=np.complex64),  # sigma_y
        csr_matrix([[1, 0], [0, -1]], dtype=np.complex64),  # sigma_z
    )

    # Create initial Majorana operators
    majorana_0 = pauli[:2]
    majorana_c0 = pauli[2]

    # With loop, build Majorana operators from the initial ones
    for i in range(N // 2 - 1):
        # Create identity matrix corresponding to old Majorana operators
        eye_mat = eye_array(2 ** (i + 1), format="csr", dtype=np.complex64)

        # Initialize new Majorana operators
        majorana = [None for _ in range(len(majorana_0) + 2)]

        # Create new Majorana operators
        for j in range(len(majorana_0)):
            majorana[j] = kron(pauli[0], majorana_0[j], format="csr")
        majorana[-2] = kron(pauli[0], majorana_c0, format="csr")
        majorana[-1] = kron(pauli[1], eye_mat, format="csr")

        # If not the last Majorana operators, update new as old
        if i < N // 2 - 2:
            majorana_0 = majorana
            majorana_c0 = kron(pauli[2], eye_mat, format="csr")
        # Else, return the last Majorana operators
        else:
            return tuple(majorana)


# ----------------------------------------
# Construct Products of Pairs of Majoranas
# ----------------------------------------
def create_majorana_pairs(
    N: int | None = None,
    majorana: tuple[csr_matrix, ...] | None = None,
) -> tuple[tuple[csr_matrix, ...], ...]:
    """Create all ψ_j * ψ_k (j < k) pairs of Majorana fermion operators."""
    # If majorana is None, create Majorana operators
    if majorana is None:
        # Check if N are provided
        if N is not None:
            # Create Majorana operators
            majorana = create_majoranas(N)
        else:
            # Raise error if N is not provided
            raise ValueError("N must be provided if majorana is None.")
    else:
        # Check if majorana is a list of Majorana operators
        if not isinstance(majorana, tuple) or not all(
            isinstance(m, csr_matrix) for m in majorana
        ):
            # Raise error if majorana is not a tuple of Majorana operators
            raise ValueError("majorana must be a tuple of Majorana operators.")

        # Store number of Majorana operators
        N = len(majorana)

    # Initialize nested list of Majorana pairs
    pairs = [[None for _ in range(N)] for _ in range(N)]

    # Fill upper triangle of Majorana pairs
    for i in range(N):
        for j in range(i + 1, N):
            pairs[i][j] = majorana[i].dot(majorana[j])

    # Return Majorana pairs
    return tuple(tuple(row) for row in pairs)
