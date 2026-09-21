from functools import cached_property
from typing import cast, override

import attrs
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_array

from ..ensembles.many_body_ensemble import (
    HermitianMatrix,
    OrthogonalMatrix,
    UnitaryMatrix,
)
from ..ensembles.syk_model import SachdevYeKitaevEnsemble
from ..fermions import DecomposedSparseArray, build_decomposed_width_matrix
from .base_compound import CompoundEnsemble


def _build_conjugated_coupling_matrix(compound: SYKCompoundEnsemble) -> csc_array:
    syk = compound.ensemble
    if syk.is_even_parity != (compound.num_free_complex_fermions % 2 == 0):
        raise ValueError("SYK model and `num_free_complex_fermions` must share parity.")

    return syk.majorana_fermion_basis.build_conjugated_compound_coupling_matrix(
        num_free_complex_fermions=compound.num_free_complex_fermions,
        coupling_strengths=compound.couplings,
        dyson_index=syk.dyson_index,
    )


@attrs.frozen(kw_only=True, eq=False, weakref_slot=True)
class SYKCompoundEnsemble(CompoundEnsemble):
    ensemble: SachdevYeKitaevEnsemble = attrs.field(
        converter=SachdevYeKitaevEnsemble.create,
        validator=attrs.validators.instance_of(SachdevYeKitaevEnsemble),
    )

    coupling_matrix_conj: csc_array = attrs.field(
        default=attrs.Factory(_build_conjugated_coupling_matrix, takes_self=True),
        init=False,
        repr=False,
    )

    @cached_property
    def decomposed_width_matrix(self) -> DecomposedSparseArray:
        return build_decomposed_width_matrix(self.coupling_matrix_conj)

    @override
    def _add_width_matrix_to_hamiltonian(self, hamiltonian: HermitianMatrix) -> None:
        row_idxs = cast(NDArray[np.int32], self.decomposed_width_matrix[0][0])
        col_idxs = cast(NDArray[np.int32], self.decomposed_width_matrix[0][1])
        width_matrix_data = self.decomposed_width_matrix[1]

        hamiltonian[row_idxs, col_idxs] -= 0.5j * width_matrix_data

    @override
    def _rotate_coupling_matrix_by_eigvecs(
        self,
        eigvecs: OrthogonalMatrix | UnitaryMatrix,
    ) -> tuple[
        NDArray[np.floating] | NDArray[np.complexfloating],
        NDArray[np.floating] | NDArray[np.complexfloating],
    ]:
        rotated_coupling_matrix_conj = self.coupling_matrix_conj @ eigvecs
        rotated_coupling_matrix = np.conjugate(
            rotated_coupling_matrix_conj.T, out=eigvecs[:, : self.num_channels]
        )
        return rotated_coupling_matrix, rotated_coupling_matrix_conj.T
