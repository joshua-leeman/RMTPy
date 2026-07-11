from __future__ import annotations

import attrs
import numpy as np
from scipy.sparse import csc_array

import rmtpy.ensembles
import rmtpy.fermions

from .base import Compound


def create_conjugated_coupling_matrix(compound: Compound) -> csc_array:
    if not isinstance(compound.ensemble, rmtpy.ensembles.SachdevYeKitaevEnsemble):
        raise TypeError("`ensemble` must be an instance of `SachdevYeKitaevEnsemble`.")
    if compound.ensemble.is_even_parity != (compound.num_free_complex_fermions % 2 == 0):
        raise ValueError("SYK model and `num_free_complex_fermions` must share parity.")

    syk = compound.ensemble
    return syk.majorana_fermion_basis.create_conjugated_compound_coupling_matrix(
        num_free_complex_fermions=compound.num_free_complex_fermions,
        coupling_strengths=compound.coupling_strengths,
        dyson_index=syk.dyson_index,
    )


@attrs.frozen(kw_only=True, eq=False, weakref_slot=True)
class SYKCompound(Compound):
    coupling_matrix_conj: csc_array = attrs.field(
        default=attrs.Factory(create_conjugated_coupling_matrix, takes_self=True),
        init=False,
        repr=False,
    )

    _decomposed_width_matrix: rmtpy.fermions.DecomposedSparseArray | None = attrs.field(
        default=None,
        init=False,
        repr=False,
    )

    @property
    def decomposed_width_matrix(self) -> rmtpy.fermions.DecomposedSparseArray:
        if self._decomposed_width_matrix is None:
            object.__setattr__(
                self,
                "_decomposed_width_matrix",
                rmtpy.fermions.create_decomposed_width_matrix(self.coupling_matrix_conj),
            )

        return self._decomposed_width_matrix

    def add_width_matrix_to_hamiltonian(self, hamiltonian: np.ndarray) -> None:
        row_idxs = self.decomposed_width_matrix[0][0]
        col_idxs = self.decomposed_width_matrix[0][1]
        width_matrix_data = self.decomposed_width_matrix[1]

        hamiltonian[row_idxs, col_idxs] -= 0.5j * width_matrix_data

    def rotate_coupling_matrix_by_eigvecs(
        self, eigvecs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        rotated_coupling_matrix_conj = self.coupling_matrix_conj @ eigvecs
        rotated_coupling_matrix = np.conjugate(
            rotated_coupling_matrix_conj.T, out=eigvecs[:, : self.num_channels]
        )
        return rotated_coupling_matrix, rotated_coupling_matrix_conj.T
