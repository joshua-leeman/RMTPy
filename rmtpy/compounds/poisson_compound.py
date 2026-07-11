from __future__ import annotations

from collections.abc import Iterator

import attrs
import numpy as np

import rmtpy.ensembles

from .base import Compound


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class PoissonCompound(Compound):
    def __attrs_post_init__(self) -> None:
        if not isinstance(self.ensemble, rmtpy.ensembles.PoissonEnsemble):
            raise TypeError("`ensemble` must be an instance of PoissonEnsemble.")

        super().__attrs_post_init__()

    def generate_effective_hamiltonian(self) -> np.ndarray:
        lapack_heev = self.ensemble._pick_lapack_heev(use_complex_dtype=True)
        blas_gemm = self.ensemble._pick_blas_gemm(use_complex_dtype=True)

        eigvecs = lapack_heev(
            self.ensemble.eigvecs_ensemble.generate_matrix(use_complex_dtype=True),
            compute_v=1,
            overwrite_a=True,
        )[1]

        rotated_coupling_matrix, _ = self.rotate_coupling_matrix_by_eigvecs(eigvecs)

        effective_hamiltonian = blas_gemm(
            alpha=-0.5j,
            a=rotated_coupling_matrix,
            trans_a=0,
            b=rotated_coupling_matrix,
            trans_b=2,
            beta=0.0,
            c=eigvecs,
            overwrite_c=True,
        )

        eigvals = self.ensemble.generate_eigenvalues()
        effective_hamiltonian[np.diag_indices(self.ensemble.dimension)] += eigvals

        return effective_hamiltonian

    def effective_hamiltonian_stream(self, realizs: int) -> Iterator[np.ndarray]:
        blas_gemm = self.ensemble._pick_blas_gemm(use_complex_dtype=True)

        for eigvals, eigvecs in self.ensemble.eigsys_stream(
            realizs, use_complex_dtype=True
        ):
            rotated_coupling_matrix, _ = self.rotate_coupling_matrix_by_eigvecs(eigvecs)

            effective_hamiltonian = blas_gemm(
                alpha=-0.5j,
                a=rotated_coupling_matrix,
                trans_a=0,
                b=rotated_coupling_matrix,
                trans_b=2,
                beta=0.0,
                c=eigvecs,
                overwrite_c=True,
            )

            effective_hamiltonian[np.diag_indices(self.ensemble.dimension)] += eigvals

            yield effective_hamiltonian
