from collections.abc import Iterator

import attrs
import numpy as np

import rmtpy.ensembles
from .compound import Compound


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class PoissonCompound(Compound):
    def __attrs_post_init__(self) -> None:
        if not isinstance(self.ensemble, rmtpy.ensembles.PoissonEnsemble):
            raise TypeError("`ensemble` must be an instance of PoissonEnsemble.")

    def generate_effective_hamiltonian(self) -> np.ndarray:
        lapack_heev: type = self.ensemble._pick_lapack_heev(use_complex_dtype=True)
        blas_gemm: type = self.ensemble._pick_blas_gemm(use_complex_dtype=True)

        eigvecs: np.ndarray = lapack_heev(
            self.ensemble.eigvecs_ensemble.generate_matrix(use_complex_dtype=True),
            compute_v=1,
            overwrite_a=True,
        )[1]

        coupling_matrix: np.ndarray = eigvecs[:, : self.num_channels].copy(order="F")
        coupling_matrix *= self.channel_coupling_strengths[None, :]

        effective_hamiltonian: np.ndarray = blas_gemm(
            alpha=-0.5j,
            a=coupling_matrix,
            trans_a=0,
            b=coupling_matrix,
            trans_b=2,
            beta=0.0,
            c=eigvecs,
            overwrite_c=True,
        )

        eigvals: np.ndarray = self.ensemble.rng.random(
            self.ensemble.dimension, self.ensemble.real_dtype
        )
        eigvals -= 0.5
        eigvals *= self.ensemble.std_dev

        effective_hamiltonian[np.diag_indices(self.ensemble.dimension)] += eigvals
        return effective_hamiltonian

    def effective_hamiltonian_stream(self, realizs: int) -> Iterator[np.ndarray]:
        blas_copy: type = self.ensemble._pick_blas_copy(use_complex_dtype=True)
        blas_gemm: type = self.ensemble._pick_blas_gemm(use_complex_dtype=True)

        coupling_matrix: np.ndarray = np.empty(
            (self.ensemble.dimension, self.num_channels),
            dtype=self.ensemble.complex_dtype,
            order="F",
        )

        for eigvals, eigvecs in self.ensemble.eigsys_stream(
            realizs, use_complex_dtype=True
        ):
            blas_copy(eigvecs[:, : self.num_channels], coupling_matrix)
            coupling_matrix *= self.channel_coupling_strengths[None, :]

            effective_hamiltonian: np.ndarray = blas_gemm(
                alpha=-0.5j,
                a=coupling_matrix,
                trans_a=0,
                b=coupling_matrix,
                trans_b=2,
                beta=0.0,
                c=eigvecs,
                overwrite_c=True,
            )

            effective_hamiltonian[np.diag_indices(self.ensemble.dimension)] += eigvals
            yield effective_hamiltonian
