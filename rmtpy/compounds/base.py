from __future__ import annotations

import inspect
import math
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import attrs
import numpy as np
from scipy.linalg import solve
from scipy.special import jn_zeros

import rmtpy.conversion
import rmtpy.density
import rmtpy.universal
from rmtpy.conversion import RMT_CONVERTER
from rmtpy.ensembles import EnsembleLike, RandomMatrixEnsemble

MAX_SPECTRAL_POLYNOMIAL_DEGREE_METADATA: dict[str, str] = {
    "dir_name": "max_polydeg",
}

NUM_FREE_COMPLEX_FERMIONS: int = 1
NUM_FREE_COMPLEX_FERMIONS_METADATA: dict[str, str] = {
    "dir_name": "Nf",
    "latex_name": r"N_\text{f}",
}

REGISTRY: dict[str, type[Compound]] = {}


def compute_default_coupling_strengths(compound: Compound) -> float:
    return np.sqrt(compound.ensemble.spectral_radius)


def compute_number_of_open_channels(compound: Compound) -> int:
    return math.comb(
        compound.ensemble.num_majoranas // 2, compound.num_free_complex_fermions
    )


def create_quantum_chaotic_compound(**kwargs: Any) -> Compound:
    return Compound.create(kwargs)


def is_num_free_fermions_valid(compound: Compound, _, num_free_fermions: int) -> None:
    if num_free_fermions > compound.ensemble.num_majoranas // 2:
        raise ValueError(
            "Number of free complex fermions must be less than the implied number "
            "of complex fermions in the quasi-stable space "
            f"{compound.ensemble.num_majoranas // 2}, got {num_free_fermions} "
            "instead."
        )


def is_num_channels_valid(compound: Compound, _, num_channels: int) -> None:
    if num_channels > compound.ensemble.dimension:
        raise ValueError(
            "Number of open channels cannot exceed the ensemble dimension; "
            f"got {num_channels} channels for dimension {compound.ensemble.dimension}."
        )


def normalize_coupling_strengths(
    coupling_strengths: Any, compound: Compound
) -> np.ndarray:
    if not isinstance(coupling_strengths, (int, float, Sequence, np.ndarray)):
        raise TypeError(
            f"Coupling strengths must be a scalar or a Sequence, "
            f"got {type(coupling_strengths).__name__}."
        )

    if isinstance(coupling_strengths, (int, float)):
        if coupling_strengths <= 0 or not np.isfinite(coupling_strengths):
            raise ValueError("Coupling strength must be a positive, finite scalar.")
        coupling_strengths_array = np.full(compound.num_channels, coupling_strengths)
        coupling_strengths_array.flags.writeable = False
        return coupling_strengths_array

    coupling_strengths_array = np.array(coupling_strengths, copy=True, order="C")
    if coupling_strengths_array.shape != (compound.num_channels,):
        raise ValueError(
            f"Coupling strengths array must have shape ({compound.num_channels},), "
            f"got {coupling_strengths_array.shape}."
        )
    if not np.isrealobj(coupling_strengths_array):
        raise ValueError("Coupling strengths array must have real, nonnegative values.")

    try:
        is_finite = np.all(np.isfinite(coupling_strengths_array))
        is_nonnegative = np.all(coupling_strengths_array >= 0)
    except TypeError as error:
        raise TypeError("Coupling strengths array must contain real numbers.") from error

    if not is_finite or not is_nonnegative:
        raise ValueError(
            "Coupling strengths array must have finite, real, nonnegative values."
        )

    coupling_strengths_array.flags.writeable = False
    return coupling_strengths_array


def register_compound_class(comp_cls: type[Compound]) -> type[Compound]:
    RMT_CONVERTER.register_structure_hook(comp_cls, structure_hook_for_compound)
    RMT_CONVERTER.register_unstructure_hook(comp_cls, unstructure_hook_for_compound)

    key = rmtpy.conversion.to_registry_key(comp_cls.__name__)
    REGISTRY[key] = comp_cls

    return comp_cls


def structure_hook_for_compound(src: dict[str, Any] | Compound, _) -> Compound:
    if type(src) in REGISTRY.values():
        return src

    comp_dict = rmtpy.conversion.normalize_dict(src, registry=REGISTRY)
    comp_args = comp_dict.pop("args")

    key = rmtpy.conversion.to_registry_key(comp_dict.pop("name"))
    comp_cls = REGISTRY[key]

    return comp_cls(**comp_args)


def unstructure_hook_for_compound(comp: Compound) -> dict[str, Any]:
    arguments = {
        name: RMT_CONVERTER.unstructure(getattr(comp, name))
        for name, attr in attrs.fields_dict(type(comp)).items()
        if attr.init
    }

    return {
        "name": rmtpy.conversion.to_registry_key(type(comp).__name__),
        "args": arguments,
    }


@register_compound_class
@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class Compound:
    """Open system formed by coupling one closed ensemble to decay channels."""

    ensemble: EnsembleLike = attrs.field(
        converter=RandomMatrixEnsemble.create,
    )
    num_free_complex_fermions: int = attrs.field(
        default=NUM_FREE_COMPLEX_FERMIONS,
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
        validator=is_num_channels_valid,
    )
    coupling_strengths: np.ndarray = attrs.field(
        default=attrs.Factory(compute_default_coupling_strengths, takes_self=True),
        converter=attrs.Converter(normalize_coupling_strengths, takes_self=True),
        repr=False,
    )

    resonance_density: rmtpy.density.DensityModel = attrs.field(
        default=None,
        init=False,
        repr=False,
    )

    def __attrs_post_init__(self) -> None:
        resonance_density = rmtpy.density.DensityModel(
            dimension=self.ensemble.dimension,
            support=(-self.ensemble.spectral_radius, self.ensemble.spectral_radius),
            polynomials=self.ensemble.spectral_polynomials,
            max_polynomial_degree=self.ensemble.max_spectral_polynomial_degree,
            weight_function=self.ensemble.spectral_weight,
            sample_stream=self.resonance_real_parts_stream,
        )
        object.__setattr__(self, "resonance_density", resonance_density)

    @classmethod
    def __attrs_init_subclass__(cls) -> None:
        if inspect.isabstract(cls):
            return

        key = rmtpy.conversion.to_registry_key(cls.__name__)
        REGISTRY[key] = cls

    @classmethod
    def create(cls, src: dict[str, Any] | Compound) -> Compound:
        return RMT_CONVERTER.structure(src, cls)

    @property
    def latex_name(self) -> str:
        return self.ensemble.latex_name

    @property
    def token_name(self) -> str:
        return self.ensemble.token_name

    @property
    def to_latex(self) -> str:
        ensemble_as_latex = self.ensemble.to_latex.replace(
            self.ensemble.latex_name, self.latex_name
        ).strip("$")
        return rmtpy.conversion.to_latex(self, latex_name=ensemble_as_latex)

    @property
    def to_path(self) -> Path:
        ensemble_path = self.ensemble.to_path
        root = Path(self.token_name) / Path(*ensemble_path.parts[1:])
        path = rmtpy.conversion.to_path(self, root=root)
        coupling_strengths_is_constant_array = np.all(
            self.coupling_strengths == self.coupling_strengths[0]
        )

        if not coupling_strengths_is_constant_array:
            coupling_strengths_id = rmtpy.conversion.create_hashed_id(
                self.coupling_strengths
            )
            return path / f"v_{coupling_strengths_id}"

        return path / f"v_{self.coupling_strengths[0]:.5g}".replace(".", "p")

    @property
    def rng_state(self) -> dict[str, Any]:
        return self.ensemble.rng.bit_generator.state

    def set_rng_state(self, rng_state: dict[str, Any] | None) -> None:
        if rng_state is not None:
            self.ensemble.set_rng_state(rng_state)

    def unstructure(self) -> dict[str, Any]:
        return RMT_CONVERTER.unstructure(self)

    def add_width_matrix_to_hamiltonian(self, hamiltonian: np.ndarray) -> None:
        diag_indices = np.diag_indices(self.num_channels)
        hamiltonian[diag_indices] -= 0.5j * (self.coupling_strengths**2)

    def generate_effective_hamiltonian(self) -> np.ndarray:
        hamiltonian = self.ensemble.generate_matrix(use_complex_dtype=True)
        self.add_width_matrix_to_hamiltonian(hamiltonian)
        return hamiltonian

    def effective_hamiltonian_stream(self, realizs: int) -> Iterator[np.ndarray]:
        for hamiltonian in self.ensemble.matrix_stream(realizs, use_complex_dtype=True):
            self.add_width_matrix_to_hamiltonian(hamiltonian)

            yield hamiltonian

    def resonances_stream(self, realizs: int) -> Iterator[np.ndarray]:
        lapack_geev = self.ensemble._pick_lapack_geev(use_complex_dtype=True)

        for effective_hamiltonian in self.effective_hamiltonian_stream(realizs):
            yield lapack_geev(
                effective_hamiltonian, compute_vl=0, compute_vr=0, overwrite_a=True
            )[0]

    def resonance_real_parts_stream(self, realizs: int) -> Iterator[np.ndarray]:
        for resonances in self.resonances_stream(realizs=realizs):
            yield resonances.real

    def rotate_coupling_matrix_by_eigvecs(
        self, eigvecs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        rotated_coupling_matrix = eigvecs[:, : self.num_channels].copy()
        rotated_coupling_matrix *= self.coupling_strengths[None, :]

        if np.isrealobj(rotated_coupling_matrix):
            rotated_coupling_matrix_conj = rotated_coupling_matrix
        else:
            rotated_coupling_matrix_conj = np.conjugate(rotated_coupling_matrix)

        return rotated_coupling_matrix, rotated_coupling_matrix_conj

    def partial_widths_stream(self, realizs: int) -> Iterator[np.ndarray]:
        for _, eigvecs in self.ensemble.eigsys_stream(realizs=realizs):
            rotated_coupling_matrix, rotated_coupling_matrix_conj = (
                self.rotate_coupling_matrix_by_eigvecs(eigvecs)
            )
            rotated_coupling_matrix *= rotated_coupling_matrix_conj

            yield rotated_coupling_matrix.real

    def reaction_matrix_stream(
        self, realizs: int, *, energies: np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        energies = np.asarray(energies)

        resolvent = np.empty(
            (energies.size, self.ensemble.dimension),
            self.ensemble.real_dtype.type,
            order="C",
        )
        reaction_matrix = np.empty(
            (energies.size, self.num_channels, self.num_channels),
            self.ensemble.complex_dtype.type,
            order="C",
        )

        for eigvals, eigvecs in self.ensemble.eigsys_stream(realizs=realizs):
            rotated_coupling_matrix, rotated_coupling_matrix_conj = (
                self.rotate_coupling_matrix_by_eigvecs(eigvecs)
            )

            np.subtract(energies[:, None], eigvals[None, :], out=resolvent)
            np.reciprocal(resolvent, out=resolvent)

            np.einsum(
                "da, nd, db -> nab",
                rotated_coupling_matrix_conj,
                resolvent,
                rotated_coupling_matrix,
                out=reaction_matrix,
                optimize=True,
            )
            reaction_matrix /= 2

            yield reaction_matrix, eigvals

    def reaction_matrix_pair_stream(
        self, realizs: int, *, energies: np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        energies = np.asarray(energies)

        resolvent = np.empty(
            (energies.size, self.ensemble.dimension),
            self.ensemble.real_dtype.type,
            order="C",
        )
        reaction_matrix = np.empty(
            (energies.size, self.num_channels, self.num_channels),
            self.ensemble.complex_dtype.type,
            order="C",
        )
        reaction_matrix_2 = np.empty(
            (energies.size, self.num_channels, self.num_channels),
            self.ensemble.complex_dtype.type,
            order="C",
        )

        for eigvals, eigvecs in self.ensemble.eigsys_stream(realizs=realizs):
            rotated_coupling_matrix, rotated_coupling_matrix_conj = (
                self.rotate_coupling_matrix_by_eigvecs(eigvecs)
            )

            np.subtract(energies[:, None], eigvals[None, :], out=resolvent)
            np.reciprocal(resolvent, out=resolvent)

            np.einsum(
                "da, nd, db -> nab",
                rotated_coupling_matrix_conj,
                resolvent,
                rotated_coupling_matrix,
                out=reaction_matrix,
                optimize=True,
            )
            reaction_matrix /= 2

            np.square(resolvent, out=resolvent)

            np.einsum(
                "da, nd, db -> nab",
                rotated_coupling_matrix_conj,
                resolvent,
                rotated_coupling_matrix,
                out=reaction_matrix_2,
                optimize=True,
            )
            reaction_matrix_2 /= 2

            yield reaction_matrix, reaction_matrix_2, eigvals

    def scattering_matrix_stream(
        self, realizs: int, *, energies: np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        numerator = np.empty(
            (energies.size, self.num_channels, self.num_channels),
            self.ensemble.complex_dtype.type,
            order="C",
        )

        for reaction_matrix, eigvals in self.reaction_matrix_stream(
            realizs, energies=np.asarray(energies)
        ):
            diag_indices = np.arange(self.num_channels)
            reaction_matrix *= 1j
            reaction_matrix[:, diag_indices, diag_indices] += 1

            np.conjugate(reaction_matrix.swapaxes(-1, -2), out=numerator)

            denominator = reaction_matrix
            s_matrix = solve(
                denominator,
                numerator,
                overwrite_a=True,
                overwrite_b=True,
                check_finite=False,
            )

            yield s_matrix, eigvals

    def wigner_smith_matrix_stream(
        self, realizs: int, *, energies: np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        reaction_matrix_adjoint = np.empty(
            (energies.size, self.num_channels, self.num_channels),
            self.ensemble.complex_dtype.type,
            order="C",
        )

        for (
            reaction_matrix,
            reaction_matrix_2,
            eigvals,
        ) in self.reaction_matrix_pair_stream(realizs, energies=np.asarray(energies)):
            diag_indices = np.arange(self.num_channels)
            reaction_matrix *= -1j
            reaction_matrix[:, diag_indices, diag_indices] += 1

            np.conjugate(reaction_matrix.swapaxes(-1, -2), out=reaction_matrix_adjoint)

            left_factor = solve(
                reaction_matrix,
                reaction_matrix_2,
                overwrite_a=True,
                overwrite_b=True,
                check_finite=False,
            )

            wigner_smith_matrix = solve(
                reaction_matrix_adjoint.swapaxes(-1, -2),
                left_factor.swapaxes(-1, -2),
                overwrite_a=True,
                overwrite_b=True,
                check_finite=False,
            )
            wigner_smith_matrix = (
                wigner_smith_matrix.swapaxes(-1, -2) + wigner_smith_matrix.conj()
            )

            yield wigner_smith_matrix, eigvals  # , s_matrix_diagonal

    def time_delays_stream(
        self, realizs: int, *, energies: np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for delay_matrix, eigvals in self.wigner_smith_matrix_stream(
            realizs, energies=np.asarray(energies)
        ):
            yield np.linalg.eigvalsh(delay_matrix), eigvals  # , s_matrix_diagonal

    def time_delay_pdf(self, times: np.ndarray) -> np.ndarray:
        global_mean_spacing = 2 * self.ensemble.spectral_radius / self.ensemble.dimension
        j_1_1 = float(jn_zeros(1, 1)[0])
        heisenberg_time = 2 * j_1_1 / global_mean_spacing

        return rmtpy.universal.time_delay_pdf(
            np.asarray(times),
            num_channels=self.num_channels,
            heisenberg_time=heisenberg_time,
        )
