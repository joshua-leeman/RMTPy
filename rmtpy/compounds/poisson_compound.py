from collections.abc import Iterator
from typing import cast, override

import attrs
import numpy as np
import scipy.linalg.blas
import scipy.linalg.lapack

from ..ensembles.many_body_ensemble import OrthogonalMatrix, UnitaryMatrix
from ..ensembles.poisson_ensemble import PoissonEnsemble
from .base_compound import ComplexHamiltonian, CompoundEnsemble


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class PoissonCompoundEnsemble(CompoundEnsemble):
    ensemble: PoissonEnsemble = attrs.field(
        converter=PoissonEnsemble.create,
        validator=attrs.validators.instance_of(PoissonEnsemble),
    )

    @override
    def generate_effective_hamiltonian(self) -> ComplexHamiltonian:
        lapack_heev = self._pick_lapack_heev(use_complex_dtype=True)
        eigvecs, _ = cast(
            tuple[OrthogonalMatrix | UnitaryMatrix, object],
            lapack_heev(
                self.ensemble.eigvec_ensemble.generate_matrix(use_complex_dtype=True),
                compute_v=1,
                overwrite_a=True,
            ),
        )

        rotated_coupling_matrix, _ = self._rotate_coupling_matrix_by_eigvecs(eigvecs)

        blas_gemm = self._pick_blas_gemm(use_complex_dtype=True)
        effective_hamiltonian = cast(
            np.ndarray[tuple[int, int], np.dtype[np.complexfloating]],
            blas_gemm(
                alpha=-0.5j,
                a=rotated_coupling_matrix,
                trans_a=0,
                b=rotated_coupling_matrix,
                trans_b=2,
                beta=0.0,
                c=eigvecs,
                overwrite_c=True,
            ),
        )

        eigvals = self.ensemble.generate_eigenvalues()
        effective_hamiltonian[np.diag_indices(self.ensemble.dimension)] += eigvals

        return effective_hamiltonian

    @override
    def effective_hamiltonian_stream(self, realizs: int) -> Iterator[ComplexHamiltonian]:
        blas_gemm = self._pick_blas_gemm(use_complex_dtype=True)

        for eigvals, eigvecs in self.ensemble.eigsys_stream(
            realizs, use_complex_dtype=True
        ):
            rotated_coupling_matrix, _ = self._rotate_coupling_matrix_by_eigvecs(eigvecs)

            effective_hamiltonian = cast(
                np.ndarray[tuple[int, int], np.dtype[np.complexfloating]],
                blas_gemm(
                    alpha=-0.5j,
                    a=rotated_coupling_matrix,
                    trans_a=0,
                    b=rotated_coupling_matrix,
                    trans_b=2,
                    beta=0.0,
                    c=eigvecs,
                    overwrite_c=True,
                ),
            )

            effective_hamiltonian[np.diag_indices(self.ensemble.dimension)] += eigvals

            yield effective_hamiltonian

    def _pick_blas_gemm(self, *, use_complex_dtype: bool = False):
        matrix_dtype = self._pick_linalg_dtype(use_complex_dtype=use_complex_dtype)
        return scipy.linalg.blas.get_blas_funcs("gemm", dtype=matrix_dtype)

    def _pick_lapack_heev(self, *, use_complex_dtype: bool = False):
        matrix_dtype = self._pick_linalg_dtype(use_complex_dtype=use_complex_dtype)
        routine = "heev" if np.issubdtype(matrix_dtype, np.complexfloating) else "syev"
        return scipy.linalg.lapack.get_lapack_funcs(routine, dtype=matrix_dtype)
