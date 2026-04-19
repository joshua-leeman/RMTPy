from collections.abc import Sequence
from itertools import combinations
from math import comb

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, eye_array, kron


def create_majoranas_fermions(num_majoranas: int) -> tuple[csr_matrix, ...]:
    pauli_matrices: tuple[csr_matrix, csr_matrix, csr_matrix] = [
        csr_matrix([[0, 1], [1, 0]], dtype=np.complex64),
        csr_matrix([[0, -1j], [1j, 0]], dtype=np.complex64),
        csr_matrix([[1, 0], [0, -1]], dtype=np.complex64),
    ]
    majoranas_0: list[csr_matrix] = pauli_matrices[:2]
    majoranas_c0: csr_matrix = pauli_matrices[2]
    for i in range(num_majoranas // 2 - 1):
        eye_matrix: csr_matrix = eye_array(2 ** (i + 1), format="csr")
        majoranas: list[csr_matrix | None] = [None] * (len(majoranas_0) + 2)
        for j in range(len(majoranas_0)):
            majoranas[j] = kron(pauli_matrices[0], majoranas_0[j], format="csr")
        majoranas[-2] = kron(pauli_matrices[0], majoranas_c0, format="csr")
        majoranas[-1] = kron(pauli_matrices[1], eye_matrix, format="csr")

        if i < num_majoranas // 2 - 2:
            majoranas_0 = majoranas
            majoranas_c0 = kron(pauli_matrices[2], eye_matrix, format="csr")
        else:
            return tuple(majoranas)


def resolve_majoranas(
    num_majoranas: int | None = None, majoranas: Sequence[csr_matrix] | None = None
) -> Sequence[csr_matrix]:
    if majoranas is None and num_majoranas is None:
        raise ValueError("Either majoranas or num_majoranas must be provided.")
    if majoranas is not None:
        if num_majoranas is not None:
            raise ValueError("Cannot specify both majoranas and num_majoranas.")
        if not isinstance(majoranas, Sequence):
            raise ValueError("majoranas must be a sequence of Majorana operators.")
        if not all(isinstance(maj, csr_matrix) for maj in majoranas):
            raise ValueError("All elements of majoranas must be csr_matrix instances.")
    if majoranas is None:
        majoranas: tuple[csr_matrix, ...] = create_majoranas_fermions(num_majoranas)

    return majoranas


def create_complex_fermions(
    num_majoranas: int | None = None, majoranas: Sequence[csr_matrix] | None = None
) -> tuple[tuple[csr_matrix, ...], tuple[csr_matrix, ...]]:
    majoranas: tuple[csr_matrix, ...] = resolve_majoranas(num_majoranas, majoranas)

    num_complex_fermions: int = len(majoranas) // 2
    annihilation_operators: list[csr_matrix | None] = [None] * num_complex_fermions
    creation_operators: list[csr_matrix | None] = [None] * num_complex_fermions

    for k in range(num_complex_fermions):
        annihilation_operators[k] = (majoranas[2 * k] - 1j * majoranas[2 * k + 1]) / 2
        creation_operators[k] = (majoranas[2 * k] + 1j * majoranas[2 * k + 1]) / 2
    return tuple(annihilation_operators), tuple(creation_operators)


def create_vacuum_state(
    complex_fermions: Sequence[Sequence[csr_matrix], Sequence[csr_matrix]],
) -> csr_matrix:
    num_complex_fermions: int = len(complex_fermions[0])
    dimension: int = 2**num_complex_fermions

    vacuum_projector: csr_matrix = eye_array(dimension, format="csr")
    for k in range(num_complex_fermions):
        vacuum_projector = complex_fermions[1][k].dot(vacuum_projector)
        vacuum_projector = complex_fermions[0][k].dot(vacuum_projector)

    arbitrary_state: csr_matrix = csr_matrix(np.ones((dimension, 1)))
    vacuum_state: csr_matrix = vacuum_projector.dot(arbitrary_state)

    vacuum_state_norm: complex = vacuum_state.multiply(vacuum_state.conj()).sum()
    if vacuum_state_norm != 1.0:
        vacuum_state /= np.sqrt(vacuum_state_norm)

    return vacuum_state


def create_block_slice(num_majoranas: int, parity: int = 0) -> tuple[slice, slice]:
    if parity not in (0, 1):
        raise ValueError("Parity must be either 0 (even) or 1 (odd).")

    complex_fermions: tuple[tuple[csr_matrix, ...], tuple[csr_matrix, ...]] = (
        create_complex_fermions(num_majoranas=num_majoranas)
    )
    vacuum_state: csr_matrix = create_vacuum_state(complex_fermions)
    if vacuum_state.count_nonzero() != 1:
        raise ValueError("Vacuum state must have only one nonzero entry.")
    elif not np.isclose(vacuum_state.multiply(vacuum_state.conj()).sum(), 1.0):
        raise ValueError("Vacuum state must be normalized.")

    size: int = 2 ** (num_majoranas // 2 - 1)
    nonzero_index: int = vacuum_state.nonzero()[0][0]
    block_idx: int = 0 if ((nonzero_index < size) ^ parity) else size
    return slice(block_idx, block_idx + size), slice(block_idx, block_idx + size)


def create_particle_hole_operator(majoranas: tuple[csr_matrix, ...]) -> csr_matrix:
    num_complex_fermions: int = len(majoranas) // 2
    dimension: int = 2**num_complex_fermions

    particle_hole_operator: csr_matrix = eye_array(dimension, format="csr")
    for k in range(num_complex_fermions):
        particle_hole_operator = majoranas[2 * k].dot(particle_hole_operator)
    return particle_hole_operator


def majoranas_to_real_basis(majoranas: Sequence[csr_matrix]) -> tuple[csr_matrix, ...]:
    num_majoranas: int = len(majoranas)
    particle_hole_operator: csr_matrix = create_particle_hole_operator(majoranas)

    real_majoranas: list[csr_matrix | None] = [None] * num_majoranas
    for k in range(num_majoranas):
        maj_P: csr_matrix = majoranas[k].dot(particle_hole_operator)
        P_maj: csr_matrix = particle_hole_operator.dot(majoranas[k])
        P_maj_P: csr_matrix = particle_hole_operator.dot(maj_P)
        real_majoranas[k] = (majoranas[k] + P_maj_P + 1j * (maj_P - P_maj)) / 2
    return tuple(real_majoranas)


def create_q_body_majorana_terms(
    q: int,
    parity_block: slice,
    num_majoranas: int | None = None,
    majoranas: Sequence[csr_matrix] | None = None,
    in_real_basis: bool = False,
) -> tuple[tuple[np.ndarray, ...], ...]:
    majoranas: tuple[csr_matrix, ...] = resolve_majoranas(num_majoranas, majoranas)

    num_majoranas: int = len(majoranas)
    num_terms: int = comb(num_majoranas, q)
    nonzeros: int = 2 ** (num_majoranas // 2 - 1)

    q_body_idxs: np.ndarray = np.empty((num_terms, 2, nonzeros), np.int32, order="C")
    if in_real_basis:
        majoranas = majoranas_to_real_basis(majoranas)
        q_body_data: np.ndarray = np.empty((num_terms, nonzeros), np.int8, order="C")
    else:
        q_body_data: np.ndarray = np.empty(
            (num_terms, nonzeros), np.complex64, order="C"
        )

    for term_num, idx_tuple in enumerate(combinations(range(num_majoranas), q)):
        q_body_term: csr_matrix = majoranas[idx_tuple[0]]
        for k in range(1, q):
            q_body_term = q_body_term.dot(majoranas[idx_tuple[k]])
        if in_real_basis:
            q_body_term = q_body_term.real.astype(np.int8)
        q_body_term: coo_matrix = q_body_term[parity_block].tocoo()

        q_body_idxs[term_num, 0, :] = q_body_term.row
        q_body_idxs[term_num, 1, :] = q_body_term.col
        q_body_data[term_num, :] = q_body_term.data

    return q_body_idxs, q_body_data
