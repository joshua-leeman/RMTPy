import math
from functools import cached_property
from itertools import combinations
from typing import cast

import attrs
import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from .validators import is_even_number

type MajoranaFermions = tuple[sparse.csr_array, ...]
type ComplexFermions = tuple[tuple[sparse.csr_array, ...], tuple[sparse.csr_array, ...]]
type ParityBlockSlice = tuple[slice, slice]
type DecomposedSparseArray = tuple[NDArray[np.int32], NDArray[np.int8 | np.complex64]]


def create_majorana_fermions(*, num_majoranas: int) -> MajoranaFermions:
    pauli_matrices = (
        sparse.csr_array([[0, 1], [1, 0]], dtype=np.complex64),
        sparse.csr_array([[0, -1j], [1j, 0]], dtype=np.complex64),
        sparse.csr_array([[1, 0], [0, -1]], dtype=np.complex64),
    )

    majorana_fermions = list(pauli_matrices[:2])
    chirality_matrix = pauli_matrices[-1]

    if num_majoranas == 2:
        return tuple(majorana_fermions)

    for i in range(num_majoranas // 2 - 1):
        identity_matrix = sparse.eye_array(cast(int, pow(2, i + 1)), format="csr")
        next_majorana_fermions: list[sparse.csr_array] = [
            sparse.kron(pauli_matrices[0], majorana_fermion, format="csr")
            for majorana_fermion in majorana_fermions
        ]
        next_majorana_fermions.append(
            sparse.kron(pauli_matrices[0], chirality_matrix, format="csr")
        )
        next_majorana_fermions.append(
            sparse.kron(pauli_matrices[1], identity_matrix, format="csr")
        )

        if i < num_majoranas // 2 - 2:
            majorana_fermions = next_majorana_fermions
            chirality_matrix = sparse.kron(
                pauli_matrices[2], identity_matrix, format="csr"
            )
        else:
            return tuple(next_majorana_fermions)

    raise RuntimeError("Failed to construct Majorana fermions: Invalid `num_majoranas`.")


def create_charge_conj_unitary_from_majoranas(
    majorana_fermions: MajoranaFermions,
) -> sparse.csr_array:
    dimension = cast(int, pow(2, len(majorana_fermions) // 2))
    running_product = sparse.eye_array(dimension, format="csr")

    for majorana_fermion in majorana_fermions[::2]:
        running_product = majorana_fermion.dot(running_product)

    return running_product


def rotate_majorana_fermions_to_real_basis(
    *,
    majorana_fermions: MajoranaFermions,
    charge_conj_unitary: sparse.csr_array,
) -> MajoranaFermions:
    rotated_majorana_fermions: list[sparse.csr_array] = []
    for majorana_fermion in majorana_fermions:
        maj_P = majorana_fermion.dot(charge_conj_unitary)
        P_maj = charge_conj_unitary.dot(majorana_fermion)
        P_maj_P = charge_conj_unitary.dot(maj_P)
        rotated_majorana_fermions.append(
            (majorana_fermion + P_maj_P + 1j * (maj_P - P_maj)) / 2
        )

    return tuple(rotated_majorana_fermions)


def create_complex_fermions_from_majoranas(
    majorana_fermions: MajoranaFermions,
) -> ComplexFermions:
    num_complex_fermions = len(majorana_fermions) // 2
    annihilation_operators: list[sparse.csr_array] = [
        (majorana_fermions[2 * k] - 1j * majorana_fermions[2 * k + 1]) / 2
        for k in range(num_complex_fermions)
    ]
    creation_operators: list[sparse.csr_array] = [
        (majorana_fermions[2 * k] + 1j * majorana_fermions[2 * k + 1]) / 2
        for k in range(num_complex_fermions)
    ]
    return tuple(annihilation_operators), tuple(creation_operators)


def create_vacuum_from_complex_fermions(
    complex_fermions: ComplexFermions,
) -> sparse.csr_array:
    num_complex_fermions = len(complex_fermions[0])
    dimension = cast(int, pow(2, num_complex_fermions))

    vacuum_projector = sparse.eye_array(dimension, format="csr")
    for k in range(num_complex_fermions):
        vacuum_projector = complex_fermions[1][k].dot(vacuum_projector)
        vacuum_projector = complex_fermions[0][k].dot(vacuum_projector)

    arbitrary_state = sparse.csr_array(np.ones((dimension, 1)))
    vacuum_state = vacuum_projector.dot(arbitrary_state)

    vacuum_squared_norm = cast(float, vacuum_state.multiply(vacuum_state.conj()).sum())
    if vacuum_squared_norm != 1.0:
        vacuum_state = cast(sparse.csr_array, vacuum_state / np.sqrt(vacuum_squared_norm))

    return vacuum_state


def create_vacuum_from_number_of_majoranas(num_majoranas: int) -> sparse.csr_array:
    majorana_fermions = create_majorana_fermions(num_majoranas=num_majoranas)
    complex_fermions = create_complex_fermions_from_majoranas(majorana_fermions)
    return create_vacuum_from_complex_fermions(complex_fermions)


def choose_block_slice_from_parity(
    *,
    is_even_parity: bool,
    num_majoranas: int,
) -> ParityBlockSlice:
    vacuum_state = create_vacuum_from_number_of_majoranas(num_majoranas)
    if vacuum_state.count_nonzero() != 1:
        raise ValueError("Vacuum state must have only one nonzero entry.")

    vacuum_state_norm = cast(float, vacuum_state.multiply(vacuum_state.conj()).sum())
    if not np.isclose(vacuum_state_norm, 1.0):
        raise ValueError("Vacuum state must be normalized.")

    parity_sector_dimension = vacuum_state.shape[0] // 2
    index_of_nonzero_entry = cast(int, vacuum_state.nonzero()[0][0])

    if (index_of_nonzero_entry < parity_sector_dimension) ^ (not is_even_parity):
        block_starting_idx = 0
    else:
        block_starting_idx = parity_sector_dimension

    return (
        slice(block_starting_idx, block_starting_idx + parity_sector_dimension),
        slice(block_starting_idx, block_starting_idx + parity_sector_dimension),
    )


def create_decomposed_q_monomials(
    *,
    q: int,
    majorana_fermions: MajoranaFermions,
    parity_block_slice: ParityBlockSlice,
    in_real_basis: bool,
) -> DecomposedSparseArray:
    num_majoranas = len(majorana_fermions)
    num_nonzeros = cast(int, pow(2, num_majoranas // 2 - 1))
    num_monomials = math.comb(num_majoranas, q)

    monomials_dtype = np.int8 if in_real_basis else np.complex64

    monomials_idxs = np.empty((num_monomials, 2, num_nonzeros), np.int32, order="C")
    monomials_data = np.empty((num_monomials, num_nonzeros), monomials_dtype, order="C")

    for term_num, idx_tuple in enumerate(combinations(range(num_majoranas), q)):
        q_body_term = majorana_fermions[idx_tuple[0]]
        for k in range(1, q):
            q_body_term = q_body_term.dot(majorana_fermions[idx_tuple[k]])

        if in_real_basis:
            q_body_term = q_body_term.real.astype(np.int8)

        q_body_term_coo = q_body_term[parity_block_slice].tocoo()
        monomials_idxs[term_num, 0, :] = q_body_term_coo.row
        monomials_idxs[term_num, 1, :] = q_body_term_coo.col
        monomials_data[term_num, :] = q_body_term_coo.data

    return monomials_idxs, monomials_data


def create_conjugated_compound_coupling_matrix(
    *,
    num_free_complex_fermions: int,
    coupling_strengths: NDArray[np.float64],
    creation_operators: MajoranaFermions,
    vacuum_state: sparse.csr_array,
    parity_block_slice: ParityBlockSlice,
    dyson_index: int,
    charge_conj_unitary: sparse.csr_array,
) -> sparse.csc_array:
    num_complex_fermions = len(creation_operators)
    num_channels = math.comb(num_complex_fermions, num_free_complex_fermions)

    if coupling_strengths.shape != (num_channels,):
        raise ValueError(
            "`coupling_strengths` must contain one coupling per generated "
            + f"operator string; expected shape ({num_channels},), "
            + f"got {coupling_strengths.shape}."
        )
    elif dyson_index == 4 and num_channels % 2 != 0:
        raise ValueError(
            "`dyson_index` == 4 requires an even number of open channels; "
            + f"got {num_channels}."
        )
    elif dyson_index == 4 and not np.allclose(
        coupling_strengths[::2], coupling_strengths[1::2]
    ):
        raise ValueError(
            "`dyson_index` == 4 requires equal coupling strengths within each "
            + "Kramers pair."
        )

    coupling_matrix_columns: list[sparse.csr_array] = []
    for indices in combinations(range(num_complex_fermions), num_free_complex_fermions):
        state = vacuum_state
        for index in reversed(indices):
            state = creation_operators[index].dot(state)

        if dyson_index == 1:
            state = cast(sparse.csr_array, state / np.sqrt(2))
            state += charge_conj_unitary.dot(state)

        coupling_matrix_columns.append(state[parity_block_slice[0]])

    coupling_matrix = sparse.hstack(coupling_matrix_columns, format="csr")

    if dyson_index == 4:
        coupling_strengths = np.repeat(coupling_strengths[::2], 2)

        pauli_1 = sparse.csr_array([[0, 1], [1, 0]])
        swap_adjacent_columns = sparse.kron(
            sparse.eye_array(num_channels // 2, format="csr"), pauli_1
        )
        coupling_matrix_columns_swapped = coupling_matrix.dot(swap_adjacent_columns)
        charge_conj_block = charge_conj_unitary[parity_block_slice]

        column_signs = np.ones(num_channels)
        column_signs[1::2] = -1
        coupling_matrix = coupling_matrix.dot(sparse.diags(column_signs, format="csr"))

        coupling_matrix += charge_conj_block.dot(coupling_matrix_columns_swapped)
        coupling_matrix = cast(sparse.csr_matrix, coupling_matrix / np.sqrt(2))

    coupling_matrix = coupling_matrix.multiply(coupling_strengths).tocsr()

    if np.allclose(np.imag(coupling_matrix.data), 0.0):
        coupling_matrix = coupling_matrix.real

    return cast(sparse.csc_array, coupling_matrix.transpose())


def create_decomposed_width_matrix(
    coupling_matrix_conj: sparse.csc_array,
) -> DecomposedSparseArray:
    width_matrix = coupling_matrix_conj.transpose() @ coupling_matrix_conj
    width_matrix_coo = width_matrix.tocoo()

    num_nonzeros = len(width_matrix_coo.data)
    width_matrix_idxs = np.empty((2, num_nonzeros), np.int32, order="C")
    width_matrix_dtype = np.result_type(width_matrix_coo.data.dtype, np.complex64)
    width_matrix_data = np.empty((num_nonzeros,), width_matrix_dtype, order="C")

    width_matrix_idxs[0, :] = width_matrix_coo.row
    width_matrix_idxs[1, :] = width_matrix_coo.col
    width_matrix_data[:] = width_matrix_coo.data

    return width_matrix_idxs, width_matrix_data


@attrs.frozen(kw_only=True, eq=False, slots=False)
class MajoranaFermionBasis:
    num_majoranas: int = attrs.field(
        validator=(
            attrs.validators.instance_of(int),
            attrs.validators.gt(0),
            is_even_number,
        ),
    )
    in_real_basis: bool = attrs.field(
        default=False,
        converter=attrs.converters.to_bool,
    )
    is_even_parity: bool = attrs.field(
        default=True,
        converter=attrs.converters.to_bool,
    )

    @cached_property
    def num_complex_fermions(self) -> int:
        return self.num_majoranas // 2

    @cached_property
    def dimension(self) -> int:
        return cast(int, pow(2, self.num_complex_fermions))

    @cached_property
    def majorana_fermions(self) -> MajoranaFermions:
        majorana_fermions = create_majorana_fermions(num_majoranas=self.num_majoranas)

        if self.in_real_basis:
            charge_conj_unitary = create_charge_conj_unitary_from_majoranas(
                majorana_fermions
            )
            majorana_fermions = rotate_majorana_fermions_to_real_basis(
                majorana_fermions=majorana_fermions,
                charge_conj_unitary=charge_conj_unitary,
            )

        return majorana_fermions

    @cached_property
    def parity_block_slice(self) -> ParityBlockSlice:
        return choose_block_slice_from_parity(
            is_even_parity=self.is_even_parity,
            num_majoranas=self.num_majoranas,
        )

    @cached_property
    def complex_fermions(self) -> ComplexFermions:
        return create_complex_fermions_from_majoranas(self.majorana_fermions)

    @cached_property
    def charge_conj_unitary(self) -> sparse.csr_array:
        return create_charge_conj_unitary_from_majoranas(self.majorana_fermions)

    @cached_property
    def vacuum_state(self) -> sparse.csr_array:
        return create_vacuum_from_complex_fermions(self.complex_fermions)

    def create_decomposed_q_monomials(self, *, q: int) -> DecomposedSparseArray:
        return create_decomposed_q_monomials(
            q=q,
            majorana_fermions=self.majorana_fermions,
            parity_block_slice=self.parity_block_slice,
            in_real_basis=self.in_real_basis,
        )

    def create_conjugated_compound_coupling_matrix(
        self,
        *,
        num_free_complex_fermions: int,
        coupling_strengths: NDArray[np.float64],
        dyson_index: int = 2,
    ) -> sparse.csc_array:
        return create_conjugated_compound_coupling_matrix(
            num_free_complex_fermions=num_free_complex_fermions,
            coupling_strengths=coupling_strengths,
            creation_operators=self.complex_fermions[1],
            vacuum_state=self.vacuum_state,
            parity_block_slice=self.parity_block_slice,
            dyson_index=dyson_index,
            charge_conj_unitary=self.charge_conj_unitary,
        )
