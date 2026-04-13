from __future__ import annotations

import hashlib
import inspect
import re
import math
from abc import ABC, abstractmethod
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from attrs import Converter, asdict, field, fields_dict, frozen
from attrs.validators import instance_of
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.linalg import eigvals, solve
from scipy.ndimage import gaussian_filter1d
from cattrs.dispatch import StructureHook, UnstructureHook

from ..ensembles import ManyBodyEnsemble
from ..utils import rmtpy_converter, insert_underscores, normalize_dict, to_registry_key


COMPOUND_REGISTRY: dict[str, type[Compound]] = {}
COMPOUND_STRUCTURE_HOOKS: dict[str, StructureHook] = {}
COMPOUND_UNSTRUCTURE_HOOKS: dict[str, UnstructureHook] = {}


def create_quantum_chaotic_compound(**kwargs: Any) -> Compound:
    return Compound.create(kwargs)


def _create_hashed_id(array: np.ndarray, num_hex: int = 16) -> str:
    hash_object: hashlib._Hash = hashlib.sha256()
    hash_object.update(str(array.dtype).encode())
    hash_object.update(str(array.shape).encode())
    hash_object.update(array.tobytes())
    return hash_object.hexdigest()[:num_hex]


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


def _to_full_array(value: int | float | np.ndarray, self_: Compound) -> np.ndarray:
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


@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class Compound(ABC):
    num_free_complex_fermions: int = field(
        default=1,
        converter=int,
        metadata={"dir_name": "Nf", "latex_name": r"N_\textrm{\tiny f}"},
    )

    @num_free_complex_fermions.validator
    def _num_free_complex_fermions_validator(self, _, value: int) -> None:
        num_complex_fermions: int = self.ensemble.num_majoranas // 2

        if value < 0 or value > num_complex_fermions:
            raise ValueError(
                f"Number of free complex fermions must be a nonnegative integer less than or equal to {num_complex_fermions}, got {value}."
            )

    ensemble: ManyBodyEnsemble = field(
        converter=ManyBodyEnsemble.create,
        validator=instance_of(ManyBodyEnsemble),
    )

    @ensemble.validator
    def _ensemble_validator(self, _, value: ManyBodyEnsemble) -> None:
        if inspect.isabstract(value):
            raise ValueError(f"ManyBodyEnsemble must be concrete.")

    num_channels: int = field(init=False)

    @num_channels.default
    def _default_num_channels(self) -> int:
        num_complex_fermions: int = self.ensemble.num_majoranas // 2
        return math.comb(num_complex_fermions, self.num_free_complex_fermions)

    channel_coupling_strengths: np.ndarray = field(
        repr=False,
        converter=Converter(_to_full_array, takes_self=True),
    )

    @channel_coupling_strengths.default
    def _default_channel_coupling_strengths(self) -> float:
        return np.sqrt(self.ensemble.ground_state_energy)

    _numerical_resonance_pdf: PchipInterpolator | None = field(
        init=False, default=None, repr=False
    )
    _numerical_resonance_cdf: PchipInterpolator | None = field(
        init=False, default=None, repr=False
    )
    _coupling_strengths_id: str = field(init=False, repr=False)

    @_coupling_strengths_id.default
    def _default_coupling_strengths_id(self) -> str:
        return _create_hashed_id(self.channel_coupling_strengths)

    @classmethod
    def __attrs_init_subclass__(cls) -> None:
        if not inspect.isabstract(cls):
            key: str = to_registry_key(cls.__name__)
            COMPOUND_REGISTRY[key] = cls
            COMPOUND_STRUCTURE_HOOKS[key] = rmtpy_converter.get_structure_hook(cls)
            COMPOUND_UNSTRUCTURE_HOOKS[key] = rmtpy_converter.get_unstructure_hook(cls)

    @classmethod
    def create(cls, src: dict[str, Any] | Compound) -> Compound:
        return rmtpy_converter.structure(src, cls)

    @property
    def _path_name(self) -> str:
        return insert_underscores(f"{self.ensemble._nickname}Compound")

    @property
    def _latex_name(self) -> str:
        return f"\\textrm{{{re.sub(r'_', ' ', self.ensemble._nickname + ' Compound')}}}"

    @property
    def to_path(self) -> Path:
        path: Path = Path(self._path_name)
        ensemble_path: Path = self.ensemble.to_path
        ensemble_args_path: Path = Path(*ensemble_path.parts[1:])
        path /= ensemble_args_path

        self_asdict: dict[str, Any] = asdict(self)
        for name, attr in fields_dict(type(self)).items():
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
            self.ensemble._latex_name, self._latex_name
        ).rstrip("$")

        self_asdict: dict[str, Any] = asdict(self)
        for name, attr in fields_dict(type(self)).items():
            if attr.metadata.get("latex_name") is not None:
                latex_str += rf"\ {attr.metadata['latex_name']}={self_asdict[name]}"
        return latex_str + "$"

    @property
    def rng_state(self) -> dict[str, Any]:
        return self.ensemble.rng.bit_generator.state

    def set_rng_state(self, rng_state: dict[str, Any] | None) -> None:
        if rng_state is not None:
            self.ensemble.set_rng_state(rng_state)

    def unstructure(self) -> dict[str, Any]:
        return rmtpy_converter.unstructure(self)

    def generate_effective_hamiltonian(self) -> np.ndarray:
        hamiltonian: np.ndarray = self.ensemble.generate_matrix()
        self.add_width_matrix(matrix=hamiltonian)
        return hamiltonian

    def effective_hamiltonian_stream(self, realizs: int) -> Iterator[np.ndarray]:
        for hamiltonian in self.ensemble.matrix_stream(realizs, use_complex_dtype=True):
            self.add_width_matrix(matrix=hamiltonian)
            yield hamiltonian

    def resonances_stream(self, realizs: int) -> Iterator[np.ndarray]:
        for effective_hamiltonian in self.effective_hamiltonian_stream(realizs):
            yield eigvals(effective_hamiltonian, overwrite_a=True, check_finite=False)

    def resonance_pdf(
        self,
        values: int | float | np.ndarray,
        _num_bins: int = 200,
        _factor: float = 1.2,
        _sigma: float = 2.0,
    ) -> np.ndarray:
        real_dtype: type[np.floating] = self.ensemble.real_dtype.type
        if isinstance(values, (int, float)):
            values: np.ndarray = np.array([values], dtype=real_dtype)

        if self._numerical_resonance_pdf is None:
            object.__setattr__(
                self,
                "_numerical_resonance_pdf",
                self._create_numerical_resonance_pdf(_num_bins, _factor, _sigma),
            )

        return self._numerical_resonance_pdf(values)

    def resonance_cdf(
        self,
        values: int | float | np.ndarray,
        _factor: float = 1.2,
        _num_bins: int = 200,
        _sigma: float = 2.0,
    ) -> np.ndarray:
        real_dtype: type[np.floating] = self.ensemble.real_dtype.type
        if isinstance(values, (int, float)):
            values: np.ndarray = np.array([values], dtype=real_dtype)

        if self._numerical_resonance_cdf is None:
            object.__setattr__(
                self,
                "_numerical_resonance_cdf",
                self._create_numerical_resonance_cdf(_num_bins, _factor, _sigma),
            )

        return self._numerical_resonance_cdf(values)

    def unfold(self, resonances: np.ndarray) -> np.ndarray:
        return self.ensemble.dimension * (
            self.resonance_cdf(resonances) - self.resonance_cdf(np.array([0.0]))
        )

    def unfold_locally(self, resonances: np.ndarray, widths: np.ndarray) -> np.ndarray:
        return self.ensemble.dimension * (
            self.resonance_cdf(resonances + widths / 2)
            - self.resonance_cdf(resonances - widths / 2)
        )

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

        for matrix, matrix_2 in self.reaction_matrix_pair_stream(energies, realizs):
            i: np.ndarray = np.arange(self.num_channels)
            matrix *= -1j
            matrix[:, i, i] += 1

            Q: np.ndarray = solve(
                matrix,
                matrix_2,
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

    def _create_numerical_resonance_pdf(
        self, num_bins: int = 200, factor: float = 1.2, sigma: float = 2.0
    ) -> np.ndarray:
        total_counts_per_dimension: int = 2**13 // self.ensemble.dimension
        realizs: int = max(total_counts_per_dimension, 1)

        energy_0: float = self.ensemble.ground_state_energy
        bins: np.ndarray = factor * np.linspace(-energy_0, energy_0, num_bins + 1)
        counts: np.ndarray = np.zeros(num_bins)

        for complex_energies in self.resonances_stream(realizs):
            resonances: np.ndarray = complex_energies.real
            counts[:] += np.histogram(resonances, bins=bins)[0]

        centers: np.ndarray = (bins[:-1] + bins[1:]) / 2
        histogram: np.ndarray = counts / np.sum(counts * np.diff(bins))
        smoothed_histogram: np.ndarray = gaussian_filter1d(histogram, sigma=sigma)

        return PchipInterpolator(centers, smoothed_histogram, extrapolate=True)

    def _create_numerical_resonance_cdf(
        self, num_bins: int = 200, factor: float = 1.2, sigma: float = 2.0
    ) -> np.ndarray:
        energy_0: float = self.ensemble.ground_state_energy
        energies: np.ndarray = factor * np.linspace(-energy_0, energy_0, num_bins + 1)
        pdf_vals: np.ndarray = self.resonance_pdf(energies, num_bins, factor, sigma)
        cdf_vals: np.ndarray = cumulative_trapezoid(pdf_vals, energies, initial=0)

        return PchipInterpolator(energies, cdf_vals, extrapolate=True)

    @abstractmethod
    def add_width_matrix(self, matrix: np.ndarray) -> np.ndarray:
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
    def reaction_matrix_pair_stream(
        self, energies: float | np.ndarray, realizs: int
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        pass


@rmtpy_converter.register_structure_hook
def compound_structure_hook(src: dict[str, Any] | Compound, _) -> Compound:
    if type(src) in COMPOUND_REGISTRY.values():
        return src

    compound_dict: dict[str, Any] = normalize_dict(src, COMPOUND_REGISTRY)
    compound_arguments: dict[str, Any] = compound_dict.pop("args")
    key: str = to_registry_key(compound_dict.pop("name"))
    compound_class: type[Compound] = COMPOUND_REGISTRY[key]
    compound_instance: Compound = compound_class(**compound_arguments)
    return compound_instance


@rmtpy_converter.register_unstructure_hook
def compound_unstructure_hook(
    compound_instance: Compound,
) -> dict[str, str | dict[str, Any]]:
    compound_name: str = type(compound_instance).__name__
    unstructured_compound: dict[str, Any] = asdict(compound_instance)
    unstructured_compound["name"] = to_registry_key(compound_name)
    unstructured_compound = normalize_dict(unstructured_compound, COMPOUND_REGISTRY)
    unstructured_compound["args"]["ensemble"] = rmtpy_converter.unstructure(
        compound_instance.ensemble
    )
    return unstructured_compound
