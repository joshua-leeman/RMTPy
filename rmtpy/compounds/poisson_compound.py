from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from attrs import field, frozen
from attrs.validators import instance_of

from ._compound import Compound, _to_1D_array
from ..ensembles import PoissonEnsemble, WignerDysonEnsemble


@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class PoissonCompound(Compound):
    ensemble: PoissonEnsemble = field(
        converter=PoissonEnsemble.create, validator=instance_of(PoissonEnsemble)
    )

    def generate_effective_hamiltonian(self, energy: float) -> np.ndarray:
        raise NotImplementedError(
            "Effective Hamiltonian generation is not implemented for the Poisson compound."
        )

    def effective_hamiltonian_stream(self, realizs: int) -> Iterator[np.ndarray]:
        raise NotImplementedError(
            "Effective Hamiltonian stream is not implemented for the Poisson compound."
        )

    def resonance_stream(self, realizs: int) -> Iterator[np.ndarray]:
        dimension: int = self.ensemble.dimension
        rng: np.random.Generator = self.ensemble.rng
        num_channels: int = self.num_channels
        strengths: np.ndarray = self.channel_coupling_strengths

        for eigvals in self.ensemble.eigvals_stream(realizs):
            random_idx: np.ndarray = rng.choice(dimension, num_channels, replace=False)
            yield eigvals[random_idx] - 1j * (strengths**2) / 2

    def partial_widths_stream(self, realizs: int) -> Iterator[np.ndarray]:
        eigvecs_ensemble: WignerDysonEnsemble = self.ensemble
        complex_dtype: type[np.complexfloating] = self.ensemble.dtype.type
        dimension: int = self.ensemble.dimension
        rng: np.random.Generator = self.ensemble.rng
        num_channels: int = self.num_channels
        strengths: np.ndarray = self.channel_coupling_strengths

        coupling_matrix_dagger: np.ndarray = np.empty(
            (num_channels, dimension), complex_dtype, order="C"
        )
        for _, eigvecs in eigvecs_ensemble.eigsys_stream(realizs):
            random_idx: np.ndarray = rng.choice(
                dimension, size=num_channels, replace=False
            )
            coupling_matrix_dagger[...] = eigvecs[random_idx, :]

            del eigvecs

            coupling_matrix_dagger *= (strengths**2)[:, None]
            coupling_matrix_dagger *= coupling_matrix_dagger.conj()

            yield coupling_matrix_dagger.T.real

    def reaction_matrix_stream(
        self, energies: float | np.ndarray, realizs: int
    ) -> Iterator[np.ndarray]:
        eigvecs_ensemble: WignerDysonEnsemble = self.ensemble
        complex_dtype: type[np.complexfloating] = self.ensemble.dtype.type
        real_dtype: type[np.floating] = self.ensemble.real_dtype.type
        dimension: int = self.ensemble.dimension
        rng: np.random.Generator = self.ensemble.rng
        num_channels: int = self.num_channels
        strengths: np.ndarray = self.channel_coupling_strengths

        energies: np.ndarray = _to_1D_array(energies)
        num_energies = energies.size

        resolvent: np.ndarray = np.empty(
            (num_energies, dimension), real_dtype, order="C"
        )
        coupling_matrix: np.ndarray = np.empty(
            (dimension, num_channels), complex_dtype, order="C"
        )
        coupling_matrix_dagger: np.ndarray = np.empty(
            (num_channels, dimension), complex_dtype, order="C"
        )
        reaction_matrix: np.ndarray = np.empty(
            (num_energies, num_channels, num_channels), complex_dtype, order="C"
        )

        for eigvals, eigvecs in eigvecs_ensemble.eigsys_stream(realizs):
            random_idx: np.ndarray = rng.choice(
                dimension, size=num_channels, replace=False
            )
            coupling_matrix_dagger[...] = eigvecs[random_idx, :]

            del eigvecs

            coupling_matrix_dagger *= strengths[:, None] / np.sqrt(2)
            coupling_matrix[...] = coupling_matrix_dagger.T.conj()

            np.subtract(energies[:, None], eigvals[None, :], out=resolvent)
            np.reciprocal(resolvent, out=resolvent)

            np.einsum(
                "ad, nd, db -> nab",
                coupling_matrix_dagger,
                resolvent,
                coupling_matrix,
                out=reaction_matrix,
                optimize=True,
            )

            yield reaction_matrix

    def reaction_and_reaction_2_matrices_stream(
        self, energies: float | np.ndarray, realizs: int
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        eigvecs_ensemble: WignerDysonEnsemble = self.ensemble
        complex_dtype: type[np.complexfloating] = self.ensemble.dtype.type
        real_dtype: type[np.floating] = self.ensemble.real_dtype.type
        dimension: int = self.ensemble.dimension
        rng: np.random.Generator = self.ensemble.rng
        num_channels: int = self.num_channels
        strengths: np.ndarray = self.channel_coupling_strengths

        energies: np.ndarray = _to_1D_array(energies)
        num_energies: int = energies.size

        resolvent: np.ndarray = np.empty(
            (num_energies, dimension), real_dtype, order="C"
        )
        coupling_matrix: np.ndarray = np.empty(
            (dimension, num_channels), complex_dtype, order="C"
        )
        coupling_matrix_dagger: np.ndarray = np.empty(
            (num_channels, dimension), complex_dtype, order="C"
        )
        reaction_matrix: np.ndarray = np.empty(
            (num_energies, num_channels, num_channels), complex_dtype, order="C"
        )

        reaction_matrix_2: np.ndarray = np.empty(
            (num_energies, num_channels, num_channels), complex_dtype, order="C"
        )

        for eigvals, eigvecs in eigvecs_ensemble.eigsys_stream(realizs):
            random_idx: np.ndarray = rng.choice(
                dimension, size=num_channels, replace=False
            )
            coupling_matrix_dagger[...] = eigvecs[random_idx, :]

            del eigvecs

            coupling_matrix_dagger *= strengths[:, None] / np.sqrt(2)
            coupling_matrix[...] = coupling_matrix_dagger.T.conj()

            np.subtract(energies[:, None], eigvals[None, :], out=resolvent)
            np.reciprocal(resolvent, out=resolvent)

            np.einsum(
                "ad, nd, db -> nab",
                coupling_matrix_dagger,
                resolvent,
                coupling_matrix,
                out=reaction_matrix,
                optimize=True,
            )

            np.square(resolvent, out=resolvent)

            np.einsum(
                "ad, nd, db -> nab",
                coupling_matrix_dagger,
                resolvent,
                coupling_matrix,
                out=reaction_matrix_2,
                optimize=True,
            )

            yield reaction_matrix, reaction_matrix_2
