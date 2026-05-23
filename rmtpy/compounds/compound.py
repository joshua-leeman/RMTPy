from __future__ import annotations

import inspect
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Sequence

import attrs
import numpy as np
from cattrs.dispatch import StructureHook, UnstructureHook
from scipy.linalg import solve

import rmtpy.density
import rmtpy.conversion
import rmtpy.ensembles

NUM_FREE_COMPLEX_FERMIONS_DEFAULT: int = 1
NUM_FREE_COMPLEX_FERMIONS_METADATA: dict[str, str] = {
    "dir_name": "Nf",
    "latex_name": r"N_\textrm{\tiny f}",
}

REGISTRY: dict[str, type[Compound]] = {}
STRUCTURE_HOOKS: dict[str, StructureHook] = {}
UNSTRUCTURE_HOOKS: dict[str, UnstructureHook] = {}


def create_quantum_chaotic_compound(**kwargs: Any) -> Compound:
    return Compound.create(kwargs)


def create_coupling_strengths_id(compound: Compound) -> str:
    return rmtpy.conversion.create_hashed_id(compound.channel_coupling_strengths)


def compute_default_coupling_strengths(compound: Compound) -> float:
    return np.sqrt(compound.ensemble.spectral_radius)


def compute_number_of_open_channels(compound: Compound) -> int:
    return 2**compound.num_free_complex_fermions


def normalize_coupling_strengths(strengths: Any, compound: Compound) -> np.ndarray:
    if not isinstance(strengths, (int, float, Sequence, np.ndarray)):
        raise TypeError(
            f"Coupling strengths must be a scalar or a Sequence, "
            f"got {type(strengths).__name__}."
        )

    if isinstance(strengths, (int, float)):
        if strengths <= 0 or not np.isfinite(strengths):
            raise ValueError("Coupling strength must be a positive, finite scalar.")
        return np.full(compound.num_channels, strengths)

    strengths: np.ndarray = np.ascontiguousarray(strengths)
    if strengths.shape != (compound.num_channels,):
        raise ValueError(
            f"Coupling strengths array must have shape ({compound.num_channels},), "
            f"got {strengths.shape}."
        )
    if not np.isrealobj(strengths) or np.any(strengths < 0):
        raise ValueError("Coupling strengths array must have real, nonnegative values.")

    return strengths


def is_num_free_fermions_valid(compound: Compound, _, num_free_fermions: int) -> None:
    if num_free_fermions > compound.ensemble.num_majoranas // 2:
        raise ValueError(
            f"Number of free complex fermions must be less than the implied number of complex "
            f"fermions in the quasi-stable space {compound.ensemble.num_majoranas // 2}, got "
            f"{num_free_fermions} instead."
        )


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class Compound:
    ensemble: rmtpy.ensembles.EnsembleLike = attrs.field(
        converter=rmtpy.ensembles.RandomMatrixEnsemble.create,
    )

    num_free_complex_fermions: int = attrs.field(
        default=NUM_FREE_COMPLEX_FERMIONS_DEFAULT,
        converter=int,
        validator=[
            attrs.validators.ge(0),
            is_num_free_fermions_valid,
        ],
        metadata=NUM_FREE_COMPLEX_FERMIONS_METADATA,
    )
    num_channels: int = attrs.field(
        default=attrs.Factory(compute_number_of_open_channels, takes_self=True),
        init=False,
    )

    channel_coupling_strengths: np.ndarray = attrs.field(
        default=attrs.Factory(compute_default_coupling_strengths, takes_self=True),
        converter=attrs.Converter(normalize_coupling_strengths, takes_self=True),
        repr=False,
    )

    resonance_density: rmtpy.density.DensityModel = attrs.field(
        default=None,
        init=False,
        repr=False,
    )

    _coupling_strengths_id: str = attrs.field(
        default=attrs.Factory(create_coupling_strengths_id, takes_self=True),
        init=False,
        repr=False,
    )

    def __attrs_post_init__(self) -> None:
        resonance_density: rmtpy.density.DensityModel = rmtpy.density.DensityModel(
            dimension=self.ensemble.dimension,
            support=(
                tuple([-self.ensemble.spectral_radius, self.ensemble.spectral_radius]),
            ),
            polynomials=self.ensemble.spectral_polynomials,
            max_polynomial_degree=self.ensemble.max_spectral_polynomial_degree,
            weight_function=self.ensemble.spectral_weight,
            sample_stream=self.resonances_stream,
        )
        object.__setattr__(self, "resonance_density", resonance_density)

    @classmethod
    def __attrs_init_subclass__(cls) -> None:
        if inspect.isabstract(cls):
            return

        key: str = rmtpy.conversion.to_registry_key(cls.__name__)
        REGISTRY[key] = cls
        STRUCTURE_HOOKS[key] = rmtpy.conversion.CONVERTER.get_structure_hook(cls)
        UNSTRUCTURE_HOOKS[key] = rmtpy.conversion.CONVERTER.get_unstructure_hook(cls)

    @classmethod
    def create(cls, src: dict[str, Any] | Compound) -> Compound:
        return rmtpy.conversion.CONVERTER.structure(src, cls)

    @property
    def latex_name(self) -> str:
        return self.ensemble.latex_name + r"\textrm{ Compound}"

    @property
    def token_name(self) -> str:
        return self.ensemble.token_name + "_Compound"

    @property
    def to_latex(self) -> str:
        ensemble_as_latex: str = self.ensemble.to_latex.replace(
            self.ensemble.latex_name, self.latex_name
        ).strip("$")

        return rmtpy.conversion.to_latex(self, ensemble_as_latex)

    @property
    def to_path(self) -> Path:
        ensemble_path: Path = self.ensemble.to_path
        root: Path = Path(self.token_name) / Path(*ensemble_path.parts[1:])
        path: Path = rmtpy.conversion.to_path(self, root)

        coupling_strengths_is_constant_array: bool = (
            self.channel_coupling_strengths == self.channel_coupling_strengths[0]
        )

        if not coupling_strengths_is_constant_array:
            return path / f"v_{self._coupling_strengths_id}"

        return path / f"v_{self.channel_coupling_strengths[0]:.5g}".replace(".", "p")

    @property
    def rng_state(self) -> dict[str, Any]:
        return self.ensemble.rng.bit_generator.state

    def set_rng_state(self, rng_state: dict[str, Any] | None) -> None:
        if rng_state is not None:
            self.ensemble.set_rng_state(rng_state)

    def unstructure(self) -> dict[str, Any]:
        return rmtpy.conversion.CONVERTER.unstructure(self)

    def generate_effective_hamiltonian(self) -> np.ndarray:
        diag_indices: np.ndarray = np.diag_indices(self.num_channels)
        hamiltonian: np.ndarray = self.ensemble.generate_matrix(use_complex_dtype=True)
        hamiltonian[diag_indices] -= 0.5j * (self.channel_coupling_strengths**2)
        return hamiltonian

    def effective_hamiltonian_stream(self, realizs: int) -> Iterator[np.ndarray]:
        diag_indices: np.ndarray = np.diag_indices(self.num_channels)
        for hamiltonian in self.ensemble.matrix_stream(realizs, use_complex_dtype=True):
            hamiltonian[diag_indices] -= 0.5j * (self.channel_coupling_strengths**2)
            yield hamiltonian

    def resonances_stream(self, realizs: int) -> Iterator[np.ndarray]:
        lapack_geev: type = self.ensemble._pick_lapack_geev(use_complex_dtype=True)
        for hamiltonian_eff in self.effective_hamiltonian_stream(realizs):
            yield lapack_geev(
                hamiltonian_eff, compute_vl=0, compute_vr=0, overwrite_a=True
            )[0]

    def partial_widths_stream(self, realizs: int) -> Iterator[np.ndarray]:
        for _, eigvecs in self.ensemble.eigsys_stream(realizs):
            coupling_matrix: np.ndarray = eigvecs[:, : self.num_channels]
            coupling_matrix *= self.channel_coupling_strengths[None, :]
            coupling_matrix *= coupling_matrix.conj()

            yield coupling_matrix.real

    def reaction_matrix_stream(
        self, energies: np.ndarray, realizs: int
    ) -> Iterator[np.ndarray]:
        energies = np.asarray(energies)

        resolvent: np.ndarray = np.empty(
            (energies.size, self.ensemble.dimension),
            self.ensemble.real_dtype,
            order="C",
        )
        reaction_matrix: np.ndarray = np.empty(
            (energies.size, self.num_channels, self.num_channels),
            self.ensemble.complex_dtype,
            order="C",
        )

        for eigvals, eigvecs in self.ensemble.eigsys_stream(realizs):
            coupling_matrix: np.ndarray = eigvecs[:, : self.num_channels]
            coupling_matrix *= self.channel_coupling_strengths[None, :] / np.sqrt(2)

            if np.isrealobj(coupling_matrix):
                coupling_matrix_conj: np.ndarray = coupling_matrix
            else:
                coupling_matrix_conj: np.ndarray = (
                    np.conjugate(coupling_matrix, out=eigvecs[:, -self.num_channels :]),
                )

            np.subtract(energies[:, None], eigvals[None, :], out=resolvent)
            np.reciprocal(resolvent, out=resolvent)

            np.einsum(
                "ad, nd, db -> nab",
                coupling_matrix_conj.T,
                resolvent,
                coupling_matrix,
                out=reaction_matrix,
                optimize=True,
            )

            yield reaction_matrix

    def reaction_matrix_pair_stream(
        self, energies: float | np.ndarray, realizs: int
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        energies = np.asarray(energies)

        resolvent: np.ndarray = np.empty(
            (energies.size, self.ensemble.dimension),
            self.ensemble.real_dtype,
            order="C",
        )
        reaction_matrix: np.ndarray = np.empty(
            (energies.size, self.num_channels, self.num_channels),
            self.ensemble.complex_dtype,
            order="C",
        )
        reaction_matrix_2: np.ndarray = np.empty(
            (energies.size, self.num_channels, self.num_channels),
            self.ensemble.complex_dtype,
            order="C",
        )

        for eigvals, eigvecs in self.ensemble.eigsys_stream(realizs):
            coupling_matrix: np.ndarray = eigvecs[:, : self.num_channels]
            coupling_matrix *= self.channel_coupling_strengths[None, :] / np.sqrt(2)

            if np.isrealobj(coupling_matrix):
                coupling_matrix_conj: np.ndarray = coupling_matrix
            else:
                coupling_matrix_conj: np.ndarray = (
                    np.conjugate(coupling_matrix, out=eigvecs[:, -self.num_channels :]),
                )

            np.subtract(energies[:, None], eigvals[None, :], out=resolvent)
            np.reciprocal(resolvent, out=resolvent)

            np.einsum(
                "ad, nd, db -> nab",
                coupling_matrix_conj.T,
                resolvent,
                coupling_matrix,
                out=reaction_matrix,
                optimize=True,
            )

            np.square(resolvent, out=resolvent)

            np.einsum(
                "ad, nd, db -> nab",
                coupling_matrix_conj.T,
                resolvent,
                coupling_matrix,
                out=reaction_matrix_2,
                optimize=True,
            )

            yield reaction_matrix, reaction_matrix_2

    def scattering_matrix_stream(
        self, energies: float | np.ndarray, realizs: int
    ) -> Iterator[np.ndarray]:
        energies = np.asarray(energies)

        numerator: np.ndarray = np.empty(
            (energies.size, self.num_channels, self.num_channels),
            self.ensemble.complex_dtype,
            order="C",
        )
        for reaction_matrix in self.reaction_matrix_stream(energies, realizs):
            diag_indices: np.ndarray = np.arange(self.num_channels)
            reaction_matrix *= 1j
            reaction_matrix[:, diag_indices, diag_indices] += 1

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
        energies = np.asarray(energies)

        for matrix, matrix_2 in self.reaction_matrix_pair_stream(energies, realizs):
            diag_indices: np.ndarray = np.arange(self.num_channels)
            matrix *= -1j
            matrix[:, diag_indices, diag_indices] += 1

            wigner_smith_matrix: np.ndarray = solve(
                matrix,
                matrix_2,
                overwrite_a=True,
                overwrite_b=True,
                check_finite=False,
            )
            wigner_smith_matrix += wigner_smith_matrix.swapaxes(-1, -2).conj()
            yield wigner_smith_matrix

    def time_delays_stream(
        self, energies: float | np.ndarray, realizs: int
    ) -> Iterator[np.ndarray]:
        for wigner_smith_matrix in self.wigner_smith_matrix_stream(energies, realizs):
            yield np.linalg.eigvalsh(wigner_smith_matrix)


@rmtpy.conversion.CONVERTER.register_structure_hook
def structure_hook_for_compound(src: dict[str, Any] | Compound, _) -> Compound:
    if type(src) in REGISTRY.values():
        return src

    comp_dict: dict[str, Any] = rmtpy.conversion.normalize_dict(src, REGISTRY)
    comp_args: dict[str, Any] = comp_dict.pop("args")
    key: str = rmtpy.conversion.to_registry_key(comp_dict.pop("name"))
    comp_cls: type[Compound] = REGISTRY[key]
    comp_inst: Compound = comp_cls(**comp_args)
    return comp_inst


@rmtpy.conversion.CONVERTER.register_unstructure_hook
def unstructure_hook_for_compound(comp: Compound) -> dict[str, Any]:
    args: dict[str, Any] = {}
    for name, attr in attrs.fields_dict(type(comp)).items():
        if attr.init:
            args[name] = rmtpy.conversion.CONVERTER.unstructure(getattr(comp, name))

    return {
        "name": rmtpy.conversion.to_registry_key(type(comp).__name__),
        "args": args,
    }


key: str = rmtpy.conversion.to_registry_key(Compound.__name__)
REGISTRY[key] = Compound
STRUCTURE_HOOKS[key] = rmtpy.conversion.CONVERTER.get_structure_hook(Compound)
UNSTRUCTURE_HOOKS[key] = rmtpy.conversion.CONVERTER.get_unstructure_hook(Compound)
