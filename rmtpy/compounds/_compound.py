from __future__ import annotations

import inspect
import math
from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np
from attrs import Converter, field, frozen
from attrs.validators import instance_of
from scipy.linalg import eigvals, solve

from ..ensembles import ManyBodyEnsemble


def _to_1D_array(energies: float | np.ndarray) -> np.ndarray:
    if not isinstance(energies, (int, float, np.ndarray)):
        raise TypeError(
            f"Energies must be a int, float or a numpy array, got {type(energies).__name__} instead."
        )
    if isinstance(energies, (int, float)):
        return np.array([energies], dtype=np.float64, order="C")
    if not np.isrealobj(energies):
        raise ValueError("Energies must have real values.")
    if energies.ndim != 1:
        raise ValueError(f"Energies must be a 1D array, got {energies.ndim}D array.")

    return energies.astype(np.float64)


def _to_full_array(value: float | np.ndarray, self_: Compound) -> np.ndarray:
    if isinstance(value, (int, float)):
        return np.full(self_.num_channels, value)
    if not isinstance(value, np.ndarray):
        raise TypeError(
            f"Coupling strengths must be a scalar or a numpy array, got {type(value).__name__}."
        )
    if value.shape != (self_.num_channels,):
        raise ValueError(
            f"Coupling strengths array must have shape ({self_.num_channels},), got {value.shape}."
        )
    return np.asfortranarray(value)


@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class Compound(ABC):
    ensemble: ManyBodyEnsemble = field(
        converter=ManyBodyEnsemble.create, validator=instance_of(ManyBodyEnsemble)
    )

    @ensemble.validator
    def _ensemble_validator(self, _, value: ManyBodyEnsemble) -> None:
        if inspect.isabstract(value):
            raise ValueError(f"ManyBodyEnsemble must be concrete.")

    num_free_complex_fermions: int = field(default=1, converter=int)

    @num_free_complex_fermions.validator
    def _num_free_complex_fermions_validator(self, _, value: int) -> None:
        num_complex_fermions: int = self.ensemble.num_majoranas // 2

        if value < 0 or value > num_complex_fermions:
            raise ValueError(
                f"Number of free complex fermions must be a nonnegative integer less than or equal to {num_complex_fermions}, got {value}."
            )

    num_channels: int = field(init=False)

    @num_channels.default
    def _num_channels_default(self) -> int:
        num_complex_fermions: int = self.ensemble.num_majoranas // 2
        return math.comb(num_complex_fermions, self.num_free_complex_fermions)

    channel_coupling_strengths: np.ndarray = field(
        converter=Converter(_to_full_array, takes_self=True)
    )

    @channel_coupling_strengths.default
    def _channel_coupling_strengths_default(self) -> float:
        return np.sqrt(self.ensemble.ground_state_energy)

    def generate_effective_hamiltonian(self) -> np.ndarray:
        H_eff: np.ndarray = self.ensemble.generate_matrix()
        self.generate_width_matrix(offset=H_eff)
        return H_eff

    def effective_hamiltonian_stream(self, realizs: int) -> Iterator[np.ndarray]:
        for hamiltonian in self.ensemble.matrix_stream(realizs, use_complex_dtype=True):
            self.generate_width_matrix(offset=hamiltonian)
            yield hamiltonian

    def resonance_stream(self, realizs: int) -> Iterator[np.ndarray]:
        for effective_hamiltonian in self.effective_hamiltonian_stream(realizs):
            yield eigvals(effective_hamiltonian, overwrite_a=True, check_finite=False)

    def scattering_matrix_stream(
        self, energies: float | np.ndarray, realizs: int
    ) -> Iterator[np.ndarray]:
        complex_dtype: type[np.complexfloating] = self.ensemble.dtype.type
        energies: np.ndarray = _to_1D_array(energies)
        num_energies: int = energies.size

        numerator: np.ndarray = np.empty(
            (num_energies, self.num_channels, self.num_channels),
            complex_dtype,
            order="C",
        )
        for reaction_matrix in self.reaction_matrix_stream(energies, realizs):
            i: np.ndarray = np.arange(self.num_channels)
            reaction_matrix *= 1j
            reaction_matrix[:, i, i] += 1
            np.conjugate(reaction_matrix.swapaxes(-1, -2), out=numerator)
            denominator: np.ndarray = reaction_matrix

            yield solve(
                denominator,
                numerator,
                overwrite_a=True,
                overwrite_b=True,
                check_finite=False,
            )

    def wigner_smith_matrix_stream(
        self, energies: float | np.ndarray, realizs: int
    ) -> Iterator[np.ndarray]:
        energies: np.ndarray = _to_1D_array(energies)
        for reaction_matrix, reaction_matrix_2 in self.reaction1_2_matrices_stream(
            energies, realizs
        ):
            i: np.ndarray = np.arange(self.num_channels)
            reaction_matrix *= -1j
            reaction_matrix[:, i, i] += 1

            Q: np.ndarray = solve(
                reaction_matrix,
                reaction_matrix_2,
                overwrite_a=True,
                overwrite_b=True,
                check_finite=False,
            )
            Q += Q.swapaxes(-1, -2).conj()
            yield Q

    def time_delays_stream(
        self, energies: float | np.ndarray, realizs: int
    ) -> Iterator[np.ndarray]:
        for wigner_smith_matrix in self.wigner_smith_matrix_stream(energies, realizs):
            yield np.linalg.eigvalsh(wigner_smith_matrix)

    @abstractmethod
    def generate_width_matrix(
        self, out: np.ndarray | None = None, offset: np.ndarray | None = None
    ) -> np.ndarray:
        pass

    @abstractmethod
    def partial_widths_stream(self, realizs: int) -> Iterator[np.ndarray]:
        pass

    @abstractmethod
    def reaction_matrix_stream(
        self, energies: float | np.ndarray, realizs: int
    ) -> Iterator[np.ndarray]:
        pass

    @abstractmethod
    def reaction1_2_matrices_stream(
        self, energies: float | np.ndarray, realizs: int
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        pass
