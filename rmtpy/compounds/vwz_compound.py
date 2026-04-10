from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from attrs import frozen, field
from attrs.validators import instance_of

from ._compound import Compound, _to_1D_array
from ..ensembles import WignerDysonEnsemble


@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class VWZCompound(Compound):
    ensemble: WignerDysonEnsemble = field(
        converter=WignerDysonEnsemble.create, validator=instance_of(WignerDysonEnsemble)
    )

    def add_width_matrix(self, matrix: np.ndarray) -> np.ndarray:
        ensemble: WignerDysonEnsemble = self.ensemble
        dimension: int = ensemble.dimension
        rng: np.random.Generator = ensemble.rng
        num_channels: int = self.num_channels
        strengths: np.ndarray = self.channel_coupling_strengths

        random_idx: np.ndarray = rng.choice(dimension, size=num_channels, replace=False)
        matrix[random_idx, random_idx] += -1j * (strengths**2) / 2
        return matrix

    def partial_widths_stream(self, realizs: int) -> Iterator[np.ndarray]:
        ensemble: WignerDysonEnsemble = self.ensemble
        complex_dtype: type[np.complexfloating] = ensemble.dtype.type
        dimension: int = ensemble.dimension
        rng: np.random.Generator = ensemble.rng
        num_channels: int = self.num_channels
        strengths: np.ndarray = self.channel_coupling_strengths

        coupling_matrix_dagger: np.ndarray = np.empty(
            (num_channels, dimension), complex_dtype, order="C"
        )
        for _, eigvecs in ensemble.eigsys_stream(realizs):
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
        ensemble: WignerDysonEnsemble = self.ensemble
        complex_dtype: type[np.complexfloating] = ensemble.dtype.type
        real_dtype: type[np.floating] = ensemble.real_dtype.type
        dimension: int = ensemble.dimension
        rng: np.random.Generator = ensemble.rng
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

        for eigvals, eigvecs in ensemble.eigsys_stream(realizs):
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
        ensemble: WignerDysonEnsemble = self.ensemble
        complex_dtype: type[np.complexfloating] = ensemble.dtype.type
        real_dtype: type[np.floating] = ensemble.real_dtype.type
        dimension: int = ensemble.dimension
        rng: np.random.Generator = ensemble.rng
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

        for eigvals, eigvecs in ensemble.eigsys_stream(realizs):
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
