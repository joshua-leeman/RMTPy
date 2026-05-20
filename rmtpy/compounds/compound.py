from __future__ import annotations

import inspect
import math
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Sequence

import attrs
import numpy as np
from cattrs.dispatch import StructureHook, UnstructureHook
from scipy.interpolate import PchipInterpolator
from scipy.linalg import solve

import rmtpy.conversion
import rmtpy.ensembles
import rmtpy.validators

COMPOUND_REGISTRY: dict[str, type[Compound]] = {}
COMPOUND_STRUCTURE_HOOKS: dict[str, StructureHook] = {}
COMPOUND_UNSTRUCTURE_HOOKS: dict[str, UnstructureHook] = {}


def create_quantum_chaotic_compound(**kwargs: Any) -> Compound:
    return Compound.create(kwargs)


def normalize_coupling_strengths(value: np.ndarray, self_: Compound) -> np.ndarray:
    if not isinstance(value, (int, float, Sequence, np.ndarray)):
        raise TypeError(
            f"Coupling strengths must be a scalar or a Sequence, got {type(value).__name__}."
        )

    if isinstance(value, (int, float)):
        if value <= 0 or not np.isfinite(value):
            raise ValueError("Coupling strength must be a positive, finite scalar.")
        return np.full(self_.num_channels, value)

    value: np.ndarray = np.ascontiguousarray(value)
    if value.shape != (self_.num_channels,):
        raise ValueError(
            f"Coupling strengths array must have shape ({self_.num_channels},), got {value.shape}."
        )
    if not np.isrealobj(value) or np.any(value < 0):
        raise ValueError("Coupling strengths array must have real, nonnegative values.")

    return value


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class Compound:
    num_free_complex_fermions: int = attrs.field(
        default=1,
        converter=int,
        metadata={"dir_name": "Nf", "latex_name": r"N_\textrm{\tiny f}"},
    )

    @num_free_complex_fermions.validator
    def _num_free_complex_fermions_validator(self, _, value: int) -> None:
        num_complex_fermions: int = self.ensemble.num_majoranas // 2

        if value < 0 or value > num_complex_fermions:
            raise ValueError(
                f"Number of free complex fermions must be a nonnegative integer "
                f"less than or equal to {num_complex_fermions}, got {value}."
            )

    ensemble: rmtpy.ensembles.ManyBodyEnsemble = attrs.field(
        converter=rmtpy.ensembles.ManyBodyEnsemble.create,
    )

    num_channels: int = attrs.field(init=False)

    @num_channels.default
    def _default_num_channels(self) -> int:
        num_complex_fermions: int = self.ensemble.num_majoranas // 2
        return math.comb(num_complex_fermions, self.num_free_complex_fermions)

    @num_channels.validator
    def _num_channels_validator(self, _, value: int) -> None:
        if value > self.ensemble.dimension // 2:
            raise ValueError(
                "Number of open channels cannot exceed half the dimension."
            )

    channel_coupling_strengths: np.ndarray = attrs.field(
        repr=False,
        converter=Converter(normalize_coupling_strengths, takes_self=True),
    )

    @channel_coupling_strengths.default
    def _default_channel_coupling_strengths(self) -> float:
        return np.sqrt(self.ensemble.spectral_radius)

    _coupling_strengths_id: str = attrs.field(init=False, repr=False)

    @_coupling_strengths_id.default
    def _default_coupling_strengths_id(self) -> str:
        return rmtpy.conversion.create_hashed_id(self.channel_coupling_strengths)

    resonance_polynomials: Callable[[np.ndarray, int], np.ndarray] | None = attrs.field(
        init=False, default=None, repr=False
    )
    resonance_polynomial_weight: Callable[[np.ndarray], np.ndarray] | None = (
        attrs.field(init=False, default=None, repr=False)
    )

    _average_resonance_coeffs: np.ndarray | None = attrs.field(
        init=False, default=None, repr=False
    )
    _average_resonance_coeffs_degree: int = attrs.field(
        init=False, default=0, repr=False
    )

    _average_resonance_pdf_interpolators: dict[
        tuple[int, float, float], PchipInterpolator
    ] = attrs.field(init=False, factory=dict, repr=False)
    _average_resonance_cdf_interpolators: dict[
        tuple[int, int, float, float], PchipInterpolator
    ] = attrs.field(init=False, factory=dict, repr=False)

    def __attrs_post_init__(self) -> None:
        if not isinstance(self.ensemble, rmtpy.ensembles.WignerDysonEnsemble):
            raise TypeError("The ensemble must be an instance of WignerDysonEnsemble.")

    @classmethod
    def __attrs_init_subclass__(cls) -> None:
        if not inspect.isabstract(cls):
            key: str = rmtpy.conversion.to_registry_key(cls.__name__)
            COMPOUND_REGISTRY[key] = cls
            COMPOUND_STRUCTURE_HOOKS[key] = RMTPY_CONVERTER.get_structure_hook(cls)
            COMPOUND_UNSTRUCTURE_HOOKS[key] = RMTPY_CONVERTER.get_unstructure_hook(cls)

    @classmethod
    def create(cls, src: dict[str, Any] | Compound) -> Compound:
        return RMTPY_CONVERTER.structure(src, cls)

    @property
    def path_name(self) -> str:
        return self.ensemble.path_name + "_Compound"

    @property
    def latex_name(self) -> str:
        return (
            f"\\textrm{{{re.sub(r'_', ' ', self.ensemble.initialism + ' Compound')}}}"
        )

    @property
    def to_path(self) -> Path:
        path: Path = Path(self.path_name)
        ensemble_path: Path = self.ensemble.to_path
        ensemble_args_path: Path = Path(*ensemble_path.parts[1:])
        path /= ensemble_args_path

        self_asdict: dict[str, Any] = attrs.asdict(self)
        for name, attr in attrs.fields_dict(type(self)).items():
            if attr.metadata.get("dir_name") is not None:
                val: str = re.sub(r"[^\w\-.]", "_", str(self_asdict[name]))
                path /= f"{attr.metadata['dir_name']}_{val.replace('.', 'p')}"

        if np.all(
            self.channel_coupling_strengths == self.channel_coupling_strengths[0]
        ):
            path /= f"v_{self.channel_coupling_strengths[0]:.5g}".replace(".", "p")
        else:
            path /= f"v_{self._coupling_strengths_id}"
        return path

    @property
    def to_latex(self) -> str:
        ensemble_latex_str: str = self.ensemble.to_latex
        latex_str: str = ensemble_latex_str.replace(
            self.ensemble.latex_name, self.latex_name
        ).rstrip("$")

        self_asdict: dict[str, Any] = attrs.asdict(self)
        for name, attr in attrs.fields_dict(type(self)).items():
            if attr.metadata.get("latex_name") is not None:
                latex_str += rf"\ {attr.metadata['latex_name']}={self_asdict[name]}"
        return latex_str + "$"

    @property
    def rng_state(self) -> dict[str, Any]:
        return self.ensemble.rng.bit_generator.state

    @property
    def has_resonance_polynomial_expansion(self) -> bool:
        return self.resonance_polynomials is not None

    @property
    def has_resonance_polynomial_weight(self) -> bool:
        return self.resonance_polynomial_weight is not None

    def set_rng_state(self, rng_state: dict[str, Any] | None) -> None:
        if rng_state is not None:
            self.ensemble.set_rng_state(rng_state)

    def unstructure(self) -> dict[str, Any]:
        return RMTPY_CONVERTER.unstructure(self)

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

    def compute_resonance_polynomials(
        self, energies: np.ndarray, degree: int
    ) -> np.ndarray:
        if not self.has_resonance_polynomial_expansion:
            raise NotImplementedError()

        validate_polynomial_degree(degree)
        x: np.ndarray = np.asarray(energies) / self.ensemble.spectral_radius
        return self.resonance_polynomials(x, degree)

    def compute_resonance_polynomial_weight(self, energies: np.ndarray) -> np.ndarray:
        if not self.has_resonance_polynomial_weight:
            raise NotImplementedError()

        return self.resonance_polynomial_weight(np.asarray(energies))

    def compute_resonance_polynomial_norms(self, degree: int) -> np.ndarray:
        validate_polynomial_degree(degree)
        return np.ones(degree + 1)

    def variate_resonance_coeffs(
        self, resonances: np.ndarray, degree: int
    ) -> np.ndarray:
        energies = np.asarray(resonances).real
        polynomials: np.ndarray = self.compute_resonance_polynomials(energies, degree)
        norms: np.ndarray = self.compute_resonance_polynomial_norms(degree)
        return np.mean(polynomials, axis=1) / norms

    def variate_resonance_pdf(
        self, resonances: np.ndarray, resonance_coeffs: np.ndarray
    ) -> np.ndarray:
        energies, coeffs = np.asarray(resonances).real, np.asarray(resonance_coeffs)
        if coeffs.ndim != 1 or len(coeffs) == 0:
            raise ValueError("resonance_coeffs must be a non-empty 1D array.")

        degree: int = len(coeffs) - 1
        polynomials: np.ndarray = self.compute_resonance_polynomials(energies, degree)
        polynomials *= resonance_coeffs[:, None]
        weight_function: np.ndarray = self.compute_resonance_polynomial_weight(energies)
        return weight_function * np.sum(polynomials, axis=0)

    def average_resonance_coeffs(self, degree: int) -> np.ndarray:
        validate_polynomial_degree(degree)
        cached_degree: int = self._average_resonance_coeffs_degree
        cached_average_coeffs: np.ndarray | None = self._average_resonance_coeffs

        if cached_average_coeffs is None or degree > cached_degree:
            cached_average_coeffs = self._create_average_resonance_coeffs(degree)
            object.__setattr__(self, "_average_resonance_coeffs", cached_average_coeffs)
            object.__setattr__(self, "_average_resonance_coeffs_degree", degree)

        return cached_average_coeffs[: degree + 1]

    def average_resonance_pdf(
        self,
        energies: np.ndarray,
        degree: int = 0,
        num_pts: int = 1000,
        factor: float = 1.2,
        sigma: float = 2.0,
    ) -> np.ndarray:
        real_dtype: type[np.floating] = self.ensemble.real_dtype.type

        if isinstance(energies, (int, float)):
            energies: np.ndarray = np.array([energies], dtype=real_dtype)
        else:
            energies = np.asarray(energies)

        if self.has_resonance_polynomial_expansion:
            average_coeffs: np.ndarray = self.average_resonance_coeffs(degree)
            return self.variate_resonance_pdf(energies, average_coeffs)

        key: tuple[int, float, float] = (num_pts, factor, sigma)
        if key not in self._average_resonance_pdf_interpolators:
            self._average_resonance_pdf_interpolators[key] = (
                self._create_average_resonance_pdf_interpolator(
                    num_bins=num_pts, factor=factor, sigma=sigma
                )
            )

        return self._average_resonance_pdf_interpolators[key](energies)

    def average_resonance_cdf(
        self,
        energies: np.ndarray,
        degree: int = 0,
        num_pts: int = 1000,
        factor: float = 1.2,
        sigma: float = 2.0,
    ) -> np.ndarray:
        validate_polynomial_degree(degree)
        real_dtype: type[np.floating] = self.ensemble.real_dtype.type

        if isinstance(energies, (int, float)):
            energies: np.ndarray = np.array([energies], dtype=real_dtype)
        else:
            energies = np.asarray(energies)

        key: tuple[int, int, float, float] = (degree, num_pts, factor, sigma)
        if key not in self._average_resonance_cdf_interpolators:
            self._average_resonance_cdf_interpolators[key] = (
                self._create_average_resonance_cdf_interpolator(
                    degree=degree, num_pts=num_pts, factor=factor, sigma=sigma
                )
            )

        return self._average_resonance_cdf_interpolators[key](energies)

    def unfold_with_average_resonance_pdf(
        self,
        energies: np.ndarray,
        degree: int = 0,
        num_pts: int = 1000,
        factor: float = 1.2,
        sigma: float = 2.0,
    ) -> np.ndarray:
        cdf = 0
        return unfold_with_cdf(energies, cdf, self.ensemble.dimension)

    def unfold_widths_with_average_resonance_pdf(
        self,
        widths: np.ndarray,
        energies: np.ndarray,
        degree: int = 0,
        num_pts: int = 1000,
        factor: float = 1.2,
        sigma: float = 2.0,
    ) -> np.ndarray:
        cdf = 0
        return unfold_widths_with_cdf(widths, energies, cdf, self.ensemble.dimension)

    def unfold_with_variate_resonance_pdf(
        self,
        energies: np.ndarray,
        resonance_coeffs: np.ndarray | None = None,
        resonances: np.ndarray | None = None,
        degree: int = 0,
        num_pts: int = 1000,
        factor: float = 1.2,
        sigma: float = 2.0,
    ) -> np.ndarray:
        cdf: PchipInterpolator = self.variate_resonance_cdf(
            resonance_coeffs=resonance_coeffs,
            resonances=resonances,
            degree=degree,
            num_pts=num_pts,
            factor=factor,
            sigma=sigma,
        )
        return unfold_with_cdf(energies, cdf, self.ensemble.dimension)

    def unfold_widths_with_variate_resonance_pdf(
        self,
        widths: np.ndarray,
        centers: np.ndarray,
        resonance_coeffs: np.ndarray | None = None,
        resonances: np.ndarray | None = None,
        degree: int = 0,
        num_pts: int = 1000,
        factor: float = 1.2,
        sigma: float = 2.0,
    ) -> np.ndarray:
        cdf: PchipInterpolator = self.variate_resonance_cdf(
            resonance_coeffs=resonance_coeffs,
            resonances=resonances,
            degree=degree,
            num_pts=num_pts,
            factor=factor,
            sigma=sigma,
        )
        return unfold_widths_with_cdf(widths, centers, cdf, self.ensemble.dimension)

    def reaction_matrix_stream(
        self, energies: float | np.ndarray, realizs: int
    ) -> Iterator[np.ndarray]:
        ensemble: ManyBodyEnsemble = self.ensemble
        complex_dtype: type[np.complexfloating] = ensemble.dtype.type
        real_dtype: type[np.floating] = ensemble.real_dtype.type
        dimension: int = ensemble.dimension
        num_channels: int = self.num_channels
        strengths: np.ndarray = self.channel_coupling_strengths

        energies = np.asarray(energies)
        num_energies = energies.size

        resolvent: np.ndarray = np.empty(
            (num_energies, dimension), real_dtype, order="C"
        )
        reaction_matrix: np.ndarray = np.empty(
            (num_energies, num_channels, num_channels), complex_dtype, order="C"
        )

        for eigvals, eigvecs in ensemble.eigsys_stream(realizs):
            coupling_matrix: np.ndarray = eigvecs[:, :num_channels]
            coupling_matrix *= strengths[None, :] / np.sqrt(2)

            if np.isrealobj(coupling_matrix):
                coupling_matrix_conj: np.ndarray = coupling_matrix
            else:
                coupling_matrix_conj: np.ndarray = np.conjugate(
                    coupling_matrix, out=eigvecs[:, -num_channels:]
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
        ensemble: ManyBodyEnsemble = self.ensemble
        complex_dtype: type[np.complexfloating] = ensemble.dtype.type
        real_dtype: type[np.floating] = ensemble.real_dtype.type
        dimension: int = ensemble.dimension
        num_channels: int = self.num_channels
        strengths: np.ndarray = self.channel_coupling_strengths

        energies = np.asarray(energies)
        num_energies: int = energies.size

        resolvent: np.ndarray = np.empty(
            (num_energies, dimension), real_dtype, order="C"
        )
        reaction_matrix: np.ndarray = np.empty(
            (num_energies, num_channels, num_channels), complex_dtype, order="C"
        )
        reaction_matrix_2: np.ndarray = np.empty(
            (num_energies, num_channels, num_channels), complex_dtype, order="C"
        )

        for eigvals, eigvecs in ensemble.eigsys_stream(realizs):
            coupling_matrix: np.ndarray = eigvecs[:, :num_channels]
            coupling_matrix *= strengths[None, :] / np.sqrt(2)

            if np.isrealobj(coupling_matrix):
                coupling_matrix_conj: np.ndarray = coupling_matrix
            else:
                coupling_matrix_conj: np.ndarray = np.conjugate(
                    coupling_matrix, out=eigvecs[:, -num_channels:]
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
        complex_dtype: type[np.complexfloating] = self.ensemble.dtype.type
        energies = np.asarray(energies)
        num_energies: int = energies.size

        numerator: np.ndarray = np.empty(
            (num_energies, self.num_channels, self.num_channels),
            complex_dtype,
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

    def _create_array_of_resonance_energies(
        self, num_pts: int = 1000, factor: float = 1.2
    ) -> np.ndarray:
        if num_pts < 2:
            raise ValueError("At least two resonance energy points are required.")
        if factor <= 0:
            raise ValueError("Resonance energy factor must be positive.")

        energy_0: float = self.ensemble.spectral_radius
        return create_histogram_bins(
            support=(-energy_0, energy_0), num_bins=num_pts - 1, scale=factor
        )

    def _create_array_of_resonance_bins(
        self, num_bins: int = 200, factor: float = 1.2
    ) -> np.ndarray:
        if num_bins < 1:
            raise ValueError("At least one resonance bin is required.")

        return self._create_array_of_resonance_energies(num_bins + 1, factor)

    def _create_average_resonance_coeffs(self, degree: int) -> np.ndarray:
        total_counts_per_dimension: int = 2**13 // self.ensemble.dimension
        realizs: int = max(total_counts_per_dimension, 10)

        average_coeffs: np.ndarray = np.zeros(degree + 1)
        for resonances in self.resonances_stream(realizs):
            average_coeffs += self.variate_resonance_coeffs(resonances, degree)

        average_coeffs /= realizs
        return average_coeffs

    def _create_average_resonance_pdf_interpolator(
        self, num_bins: int = 200, factor: float = 1.2, sigma: float = 2.0
    ) -> PchipInterpolator:
        total_counts_per_dimension: int = 2**13 // self.ensemble.dimension
        realizs: int = max(total_counts_per_dimension, 1)

        bins: np.ndarray = self._create_array_of_resonance_bins(num_bins, factor)
        counts: np.ndarray = np.zeros(num_bins)

        for complex_energies in self.resonances_stream(realizs):
            resonances: np.ndarray = complex_energies.real
            counts[:] += np.histogram(resonances, bins=bins)[0]

        return create_pdf_interpolator_from_histogram(
            bins=bins, counts=counts, sigma=sigma
        )

    def _create_average_resonance_cdf_interpolator(
        self,
        degree: int = 0,
        num_pts: int = 1000,
        factor: float = 1.2,
        sigma: float = 2.0,
    ) -> PchipInterpolator:
        energies: np.ndarray = self._create_array_of_resonance_energies(num_pts, factor)
        return create_cdf_interpolator_from_pdf(
            self.average_resonance_pdf,
            inputs=energies,
            degree=degree,
            num_pts=num_pts,
            factor=factor,
            sigma=sigma,
        )


@RMTPY_CONVERTER.register_structure_hook
def compound_structure_hook(src: dict[str, Any] | Compound, _) -> Compound:
    if type(src) in COMPOUND_REGISTRY.values():
        return src

    comp_dict: dict[str, Any] = rmtpy.conversion.normalize_dict(src, COMPOUND_REGISTRY)
    comp_args: dict[str, Any] = comp_dict.pop("args")
    key: str = rmtpy.conversion.to_registry_key(comp_dict.pop("name"))
    comp_cls: type[Compound] = COMPOUND_REGISTRY[key]
    comp_inst: Compound = comp_cls(**comp_args)
    return comp_inst


@RMTPY_CONVERTER.register_unstructure_hook
def compound_unstructure_hook(compound: Compound) -> dict[str, Any]:
    comp_name: str = type(compound).__name__
    unstr_comp: dict[str, Any] = attrs.asdict(compound)
    unstr_comp["name"] = rmtpy.conversion.to_registry_key(comp_name)
    unstr_comp = rmtpy.conversion.normalize_dict(unstr_comp, COMPOUND_REGISTRY)
    unstr_comp["args"]["ensemble"] = RMTPY_CONVERTER.unstructure(compound.ensemble)
    return unstr_comp


key: str = rmtpy.conversion.to_registry_key(Compound.__name__)
COMPOUND_REGISTRY[key] = Compound
COMPOUND_STRUCTURE_HOOKS[key] = RMTPY_CONVERTER.get_structure_hook(Compound)
COMPOUND_UNSTRUCTURE_HOOKS[key] = RMTPY_CONVERTER.get_unstructure_hook(Compound)
