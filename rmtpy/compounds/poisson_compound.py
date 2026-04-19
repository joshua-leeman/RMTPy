from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from attrs import frozen

from .compound import Compound
from ..ensembles import PoissonEnsemble, WignerDysonEnsemble


@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class PoissonCompound(Compound):
    def __attrs_post_init__(self) -> None:
        if not isinstance(self.ensemble, PoissonEnsemble):
            raise TypeError("The ensemble must be an instance of PoissonEnsemble.")

    def generate_effective_hamiltonian(self) -> np.ndarray:
        ensemble: PoissonEnsemble = self.ensemble
        eigvecs_ensemble: WignerDysonEnsemble = ensemble.eigvecs_ensemble
        real_dtype: type[np.floating] = ensemble.real_dtype.type
        rng: np.random.Generator = ensemble.rng
        dimension: int = ensemble.dimension
        std_dev: float = ensemble.std_dev
        num_channels: int = self.num_channels
        strengths: np.ndarray = self.channel_coupling_strengths

        lapack_heev: type = eigvecs_ensemble._pick_lapack_heev(use_complex_dtype=True)
        blas_gemm: type = eigvecs_ensemble._pick_blas_gemm(use_complex_dtype=True)

        eigvecs: np.ndarray = lapack_heev(
            eigvecs_ensemble.generate_matrix(use_complex_dtype=True),
            compute_v=1,
            overwrite_a=True,
        )[1]

        coupling_matrix: np.ndarray = eigvecs[:, :num_channels].copy(order="F")
        coupling_matrix *= strengths[None, :]

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

        eigvals: np.ndarray = rng.random(dimension, real_dtype)
        eigvals -= 0.5
        eigvals *= std_dev

        effective_hamiltonian[np.diag_indices(dimension)] += eigvals
        return effective_hamiltonian

    def effective_hamiltonian_stream(self, realizs: int) -> Iterator[np.ndarray]:
        ensemble: PoissonEnsemble = self.ensemble
        complex_dtype: type[np.complexfloating] = ensemble.dtype.type
        dimension: int = ensemble.dimension
        num_channels: int = self.num_channels
        strengths: np.ndarray = self.channel_coupling_strengths

        blas_copy: type = ensemble._pick_blas_copy(use_complex_dtype=True)
        blas_gemm: type = ensemble._pick_blas_gemm(use_complex_dtype=True)

        coupling_matrix: np.ndarray = np.empty(
            (dimension, num_channels), dtype=complex_dtype, order="F"
        )

        for eigvals, eigvecs in ensemble.eigsys_stream(realizs, use_complex_dtype=True):
            blas_copy(eigvecs[:, :num_channels], coupling_matrix)
            coupling_matrix *= strengths[None, :]

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

            effective_hamiltonian[np.diag_indices(dimension)] += eigvals
            yield effective_hamiltonian
