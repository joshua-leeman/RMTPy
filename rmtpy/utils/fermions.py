# rmtpy/utils/fermions.py

# Third-party imports
import numpy as np
from scipy.linalg import eig
from scipy.sparse import csr_matrix, eye_array, kron


# ------------------------
# Create Majorana Fermions
# ------------------------
def create_majoranas(N: int) -> tuple[csr_matrix, ...]:
    """Create list of N Majorana fermion operators."""

    # Create Pauli matrices
    pauli = (
        csr_matrix([[0, 1], [1, 0]]),  #     sigma_x
        csr_matrix([[0, -1j], [1j, 0]]),  #  sigma_y
        csr_matrix([[1, 0], [0, -1]]),  #    sigma_z
    )

    # Create initial Majorana operators
    majorana_0 = pauli[:2]
    majorana_c0 = pauli[2]

    # With loop, build Majorana operators from the initial ones
    for i in range(N // 2 - 1):
        # Create identity matrix corresponding to old Majorana operators
        eye_mat = eye_array(2 ** (i + 1), format="csr")

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


# -----------------------
# Create Complex Fermions
# -----------------------
def create_complex_fermions(
    N: int | None = None, majorana: tuple[csr_matrix, ...] | None = None
) -> tuple[tuple[csr_matrix, ...], tuple[csr_matrix, ...]]:
    """Create tuples of creation and annihilation operators for complex fermions."""

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

    # Alias number of complex fermion operators
    Nc = len(majorana) // 2

    # =================================================

    # Initialize lists of creation and annihilation operators
    annihilation = [None for _ in range(Nc)]
    creation = [None for _ in range(Nc)]

    # Create creation and annihilation operators from Majoranas
    for k in range(Nc):
        annihilation[k] = (majorana[2 * k] - 1j * majorana[2 * k + 1]) / 2
        creation[k] = (majorana[2 * k] + 1j * majorana[2 * k + 1]) / 2

    # Return tuples of creation and annihilation operators
    return tuple(annihilation), tuple(creation)


# ----------------------
# Construct Vacuum State
# ----------------------
def vacuum_state(
    complex_fermions: tuple[tuple[csr_matrix, ...], tuple[csr_matrix, ...]],
) -> csr_matrix:
    """Construct vacuum state from complex fermion operators."""

    # Alias complex fermions operators
    c = complex_fermions

    # Alias number of complex fermion operators
    Nc = len(c[0])

    # Alias dimension of total Hilbert space
    d = 2**Nc

    # =================================================

    # Construct projector onto vacuum state
    projector = eye_array(d, format="csr")
    for k in range(Nc):
        projector = c[0][k].dot(c[1][k].dot(projector))

    # Choose arbitrary state
    psi = csr_matrix(np.ones((d, 1)))

    # Project psi onto vacuum state and ensure normalization
    omega = projector.dot(psi)

    # Normalize psi if needed
    norm = omega.multiply(omega.conj()).sum()
    if norm != 1.0:
        omega /= np.sqrt(norm)

    # Return vacuum state
    return omega


# ----------------------------
# Slice of Parity Sector Block
# ----------------------------
def block_slice(N: int, parity: int = 0) -> tuple[slice, slice]:
    """Return slice of parity sector block for N Majorana fermions and given parity."""

    # Ensure parity is either 0 or 1
    if parity not in (0, 1):
        raise ValueError("Parity must be either 0 (even) or 1 (odd).")

    # Create complex fermions
    c = create_complex_fermions(N)

    # Construct vacuum state
    omega = vacuum_state(c)

    # Ensure vacuum state is normalized and has only one nonzero entry
    if omega.count_nonzero() != 1:
        raise ValueError("Vacuum state must have only one nonzero entry.")
    elif not np.isclose(omega.multiply(omega.conj()).sum(), 1.0):
        raise ValueError("Vacuum state must be normalized.")

    # Alias dimension of parity sector blocks
    d = 2 ** (N // 2 - 1)

    # =================================================

    # Get index of nonzero entry in vacuum state
    index = omega.nonzero()[0][0]

    # Determine slice of parity sector block based on index and parity
    start = d if (parity ^ (index < d)) else 0
    return slice(start, start + d), slice(start, start + d)


# ------------------------------------------------
# Construct Unitary Part of Particle-Hole Operator
# ------------------------------------------------
def particle_hole_operator(majorana: tuple[csr_matrix, ...]) -> csr_matrix:
    """Construct unitary part of particle-hole operator from Majorana operators."""

    # Alias number of complex fermion operators
    Nc = len(majorana) // 2

    # Alias dimension of total Hilbert space
    d = 2**Nc

    # =================================================

    # Construct unitary part of particle-hole operator
    P = eye_array(d, format="csr")
    for k in range(Nc):
        P = majorana[2 * k].dot(P)

    # Return unitary part of particle-hole operator
    return P


# ------------------------------
# Rotate Majoranas to Real Basis
# ------------------------------
def to_real_basis(majorana: tuple[csr_matrix, ...]) -> tuple[csr_matrix, ...]:
    """Rotate Majorana operators to real basis."""

    # Alias number of Majorana operators
    N = len(majorana)

    # =================================================

    # Construct unitary part of particle-hole operator
    P = particle_hole_operator(majorana)

    # Initialize list of Majorana operators in real basis
    real_majorana = [None for _ in range(N)]

    # Rotate Majorana operators to real basis
    for k in range(N):
        PmajP = P.dot(majorana[k].dot(P))
        commutator = majorana[k].dot(P) - P.dot(majorana[k])
        real_majorana[k] = (majorana[k] + PmajP + 1j * commutator) / 2

    # Return Majorana operators in real basis
    return tuple(real_majorana)


# ----------------------------------------
# Construct Products of Pairs of Majoranas
# ----------------------------------------
def create_majorana_pairs(
    N: int | None = None,
    majorana: tuple[csr_matrix, ...] | None = None,
    real_basis: bool = False,
) -> tuple[tuple[csr_matrix, ...], ...]:
    """Create all ψ_j * ψ_k (j < k) pairs of Majorana fermion operators."""

    # If majorana is None, create Majorana operators
    if majorana is None:
        # Check if N are provided
        if N is not None:
            # Create Majorana operators
            majorana = create_majoranas(N)

            # If real_basis is True, convert Majorana operators to real basis
            if real_basis:
                majorana = to_real_basis(majorana)
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
