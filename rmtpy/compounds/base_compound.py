import inspect
import math
from collections.abc import Callable, Iterator, Sequence
from functools import cached_property
from pathlib import Path
from typing import cast

import attrs
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import get_lapack_funcs, solve
from scipy.special import jn_zeros

from ..conversion import (
    RMT_CONVERTER,
    SourceDict,
    build_hashed_id,
    canonicalize_source_dict,
    to_key_of_registry,
    to_latex,
    to_path,
)
from ..density import DensityModel
from ..ensembles.many_body_ensemble import (
    HermitianMatrix,
    ManyBodyEnsemble,
    OrthogonalMatrix,
    RealEigenvalues,
    UnitaryMatrix,
)
from ..ensembles.poisson_ensemble import PoissonEnsemble
from ..universal import time_delay_pdf

type ComplexEigenvalues = NDArray[np.complexfloating]

MAX_SPECTRAL_POLYNOMIAL_DEGREE_METADATA: dict[str, str] = {
    "dir_name": "max_polydeg",
}

NUM_FREE_COMPLEX_FERMIONS: int = 1
NUM_FREE_COMPLEX_FERMIONS_METADATA: dict[str, str] = {
    "dir_name": "Nf",
    "latex_name": r"N_\text{f}",
}

REGISTRY: dict[str, type[attrs.AttrsInstance]] = {}


def _compute_default_couplings(compound: CompoundEnsemble) -> NDArray[np.floating]:
    uniform_strength = cast(float, np.sqrt(compound.ensemble.spectral_radius))
    couplings = np.full(compound.num_channels, uniform_strength)
    couplings.flags.writeable = False
    return couplings


def _compute_couplings_array(
    input: int
    | float
    | Sequence[int]
    | Sequence[float]
    | NDArray[np.integer]
    | NDArray[np.floating],
    compound: attrs.AttrsInstance,
) -> NDArray[np.floating]:
    if not isinstance(compound, CompoundEnsemble):
        raise TypeError(f"Expected CompoundEnsemble instance, got {type(compound)}.")

    if isinstance(input, (int, float)):
        if input <= 0 or not np.isfinite(input):
            raise ValueError("Coupling strength must be a positive, finite scalar.")

        couplings = np.full(compound.num_channels, input)

    couplings = np.array(input, copy=True, order="C")
    couplings.flags.writeable = False
    return couplings


def _compute_number_of_open_channels(compound: CompoundEnsemble) -> int:
    return math.comb(
        compound.ensemble.num_majoranas // 2, compound.num_free_complex_fermions
    )


def _is_couplings_array_valid(
    compound: CompoundEnsemble,
    _: attrs.Attribute[NDArray[np.floating]],
    couplings: NDArray[np.floating],
) -> None:
    if couplings.shape != (compound.num_channels,):
        raise ValueError(
            f"Coupling strengths array must have shape ({compound.num_channels},), "
            + f"got {couplings.shape}."
        )
    if not np.isrealobj(couplings):
        raise ValueError("Coupling strengths array must have real, nonnegative values.")

    try:
        is_finite = np.all(np.isfinite(couplings))
        is_nonnegative = np.all(couplings >= 0)
    except TypeError as error:
        raise TypeError("Coupling strengths array must contain real numbers.") from error

    if not is_finite or not is_nonnegative:
        raise ValueError(
            "Coupling strengths array must have finite, real, nonnegative values."
        )


def _is_num_free_fermions_valid(
    compound: CompoundEnsemble,
    _: object,
    num_free_fermions: int,
) -> None:
    if num_free_fermions > compound.ensemble.num_majoranas // 2:
        raise ValueError(
            "Number of free complex fermions must be less than the implied number "
            + "of complex fermions in the quasi-stable space "
            + f"{compound.ensemble.num_majoranas // 2}, got {num_free_fermions} "
            + "instead."
        )


def _is_num_channels_valid(
    compound: CompoundEnsemble,
    _: object,
    num_channels: int,
) -> None:
    if num_channels > compound.ensemble.dimension:
        raise ValueError(
            "Number of open channels cannot exceed the ensemble dimension; "
            + f"got {num_channels} channels for dimension {compound.ensemble.dimension}."
        )


def _structure_hook_for_compound(
    src: SourceDict | CompoundEnsemble,
    _: object,
) -> CompoundEnsemble:
    if isinstance(src, dict):
        compound_dict = canonicalize_source_dict(src, registry=REGISTRY)

        compound_type = compound_dict["type"]
        if not isinstance(compound_type, str):
            raise TypeError("Configuration `type` must be a string.")

        parameters = compound_dict["parameters"]
        if not isinstance(parameters, dict):
            raise TypeError("Configuration `parameters` must be a dictionary.")

        key = to_key_of_registry(compound_type)
        compound_factory = cast(Callable[..., CompoundEnsemble], REGISTRY[key])
        return compound_factory(**parameters)

    return src


def _unstructure_hook_for_compound(compound: CompoundEnsemble) -> SourceDict:
    fields = cast(dict[str, attrs.Attribute[object]], attrs.fields_dict(type(compound)))
    parameters = {
        name: RMT_CONVERTER.unstructure(getattr(compound, name))
        for name, attr in fields.items()
        if attr.init
    }

    return {
        "type": type(compound).__name__,
        "parameters": parameters,
    }


def _register_compound_hooks(
    compound_cls: type[CompoundEnsemble],
) -> type[CompoundEnsemble]:
    RMT_CONVERTER.register_structure_hook(compound_cls, _structure_hook_for_compound)
    RMT_CONVERTER.register_unstructure_hook(compound_cls, _unstructure_hook_for_compound)

    return compound_cls


@_register_compound_hooks
@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class CompoundEnsemble:
    ensemble: ManyBodyEnsemble = attrs.field(
        converter=ManyBodyEnsemble.create,
        validator=(
            attrs.validators.instance_of(ManyBodyEnsemble),
            attrs.validators.not_(attrs.validators.instance_of(PoissonEnsemble)),
        ),
    )
    num_free_complex_fermions: int = attrs.field(
        default=NUM_FREE_COMPLEX_FERMIONS,
        converter=int,
        validator=(
            attrs.validators.ge(0),
            _is_num_free_fermions_valid,
        ),
        metadata=NUM_FREE_COMPLEX_FERMIONS_METADATA,
    )
    num_channels: int = attrs.field(
        default=attrs.Factory(_compute_number_of_open_channels, takes_self=True),
        init=False,
        validator=_is_num_channels_valid,
    )
    couplings: NDArray[np.floating] = attrs.field(
        default=attrs.Factory(_compute_default_couplings, takes_self=True),
        converter=attrs.Converter(_compute_couplings_array, takes_self=True),
        validator=_is_couplings_array_valid,
        repr=False,
    )

    @classmethod
    def __attrs_init_subclass__(cls) -> None:
        if inspect.isabstract(cls):
            return

        key = to_key_of_registry(cls.__name__)
        REGISTRY[key] = cls

    @classmethod
    def create(cls, src: SourceDict | CompoundEnsemble) -> CompoundEnsemble:
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
        return to_latex(self, latex_name=ensemble_as_latex)

    @property
    def to_path(self) -> Path:
        ensemble_path = self.ensemble.to_path
        root = Path(self.token_name) / Path(*ensemble_path.parts[1:])
        path = to_path(self, root=root)

        couplings_is_constant_array = np.all(
            cast(NDArray[np.bool_], self.couplings == self.couplings[0])
        )
        if not couplings_is_constant_array:
            coupling_strengths_id = build_hashed_id(self.couplings)
            return path / f"v_{coupling_strengths_id}"

        return path / f"v_{self.couplings[0]:.5g}".replace(".", "p")

    @property
    def rng_state(self) -> dict[str, object]:
        return dict(self.ensemble.rng.bit_generator.state)

    @cached_property
    def resonance_density(self) -> DensityModel:
        return DensityModel(
            dimension=self.ensemble.dimension,
            support=(-self.ensemble.spectral_radius, self.ensemble.spectral_radius),
            polynomials=self.ensemble.assign_spectral_polynomials(),
            max_polynomial_degree=self.ensemble.max_spectral_polynomial_degree,
            weight_function=self.ensemble.assign_spectral_weight(),
            sample_stream=self.resonance_real_parts_stream,
        )

    def generate_effective_hamiltonian(self) -> NDArray[np.complexfloating]:
        hamiltonian = self.ensemble.generate_matrix(use_complex_dtype=True)
        self._add_width_matrix_to_hamiltonian(cast(HermitianMatrix, hamiltonian))
        return cast(NDArray[np.complexfloating], hamiltonian)

    def effective_hamiltonian_stream(
        self,
        realizs: int,
    ) -> Iterator[NDArray[np.complexfloating]]:
        for hamiltonian in self.ensemble.matrix_stream(realizs, use_complex_dtype=True):
            self._add_width_matrix_to_hamiltonian(cast(HermitianMatrix, hamiltonian))

            yield cast(NDArray[np.complexfloating], hamiltonian)

    def resonances_stream(self, realizs: int) -> Iterator[ComplexEigenvalues]:
        lapack_geev = self._pick_lapack_geev(use_complex_dtype=True)

        for effective_hamiltonian in self.effective_hamiltonian_stream(realizs):
            resonances, _ = cast(
                tuple[ComplexEigenvalues, object],
                lapack_geev(
                    effective_hamiltonian, compute_vl=0, compute_vr=0, overwrite_a=True
                ),
            )
            yield resonances

    def resonance_real_parts_stream(self, realizs: int) -> Iterator[RealEigenvalues]:
        for resonances in self.resonances_stream(realizs=realizs):
            yield np.real(resonances)

    def partial_widths_stream(self, realizs: int) -> Iterator[NDArray[np.floating]]:
        for _, eigvecs in self.ensemble.eigsys_stream(realizs=realizs):
            rotated_coupling_matrix, rotated_coupling_matrix_conj = (
                self._rotate_coupling_matrix_by_eigvecs(eigvecs)
            )
            rotated_coupling_matrix *= rotated_coupling_matrix_conj

            yield np.real(rotated_coupling_matrix)

    def reaction_matrix_stream(
        self,
        realizs: int,
        *,
        energies: NDArray[np.float64],
    ) -> Iterator[tuple[NDArray[np.complexfloating], RealEigenvalues]]:
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
                self._rotate_coupling_matrix_by_eigvecs(eigvecs)
            )

            _ = np.subtract(energies[:, None], eigvals[None, :], out=resolvent)
            _ = np.reciprocal(resolvent, out=resolvent)

            _ = np.einsum(
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
        self,
        realizs: int,
        *,
        energies: NDArray[np.float64],
    ) -> Iterator[
        tuple[NDArray[np.complexfloating], NDArray[np.complexfloating], RealEigenvalues]
    ]:
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
                self._rotate_coupling_matrix_by_eigvecs(eigvecs)
            )

            _ = np.subtract(energies[:, None], eigvals[None, :], out=resolvent)
            _ = np.reciprocal(resolvent, out=resolvent)

            _ = np.einsum(
                "da, nd, db -> nab",
                rotated_coupling_matrix_conj,
                resolvent,
                rotated_coupling_matrix,
                out=reaction_matrix,
                optimize=True,
            )
            reaction_matrix /= 2

            _ = np.square(resolvent, out=resolvent)

            _ = np.einsum(
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
        self,
        realizs: int,
        *,
        energies: NDArray[np.float64],
    ) -> Iterator[tuple[NDArray[np.complexfloating], RealEigenvalues]]:
        numerator = np.empty(
            (energies.size, self.num_channels, self.num_channels),
            self.ensemble.complex_dtype.type,
            order="C",
        )

        for reaction_matrix, eigvals in self.reaction_matrix_stream(
            realizs, energies=energies
        ):
            diag_indices = np.arange(self.num_channels)
            reaction_matrix *= 1j
            reaction_matrix[:, diag_indices, diag_indices] += 1

            _ = np.conjugate(reaction_matrix.swapaxes(-1, -2), out=numerator)

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
        self,
        realizs: int,
        *,
        energies: NDArray[np.float64],
    ) -> Iterator[tuple[NDArray[np.complexfloating], RealEigenvalues]]:
        reaction_matrix_adjoint = np.empty(
            (energies.size, self.num_channels, self.num_channels),
            self.ensemble.complex_dtype.type,
            order="C",
        )

        for (
            reaction_matrix,
            reaction_matrix_2,
            eigvals,
        ) in self.reaction_matrix_pair_stream(realizs, energies=energies):
            diag_indices = np.arange(self.num_channels)
            reaction_matrix *= -1j
            reaction_matrix[:, diag_indices, diag_indices] += 1

            _ = np.conjugate(
                reaction_matrix.swapaxes(-1, -2), out=reaction_matrix_adjoint
            )

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

            yield wigner_smith_matrix, eigvals

    def time_delays_stream(
        self,
        realizs: int,
        *,
        energies: NDArray[np.float64],
    ) -> Iterator[tuple[RealEigenvalues, RealEigenvalues]]:
        for delay_matrix, eigvals in self.wigner_smith_matrix_stream(
            realizs, energies=energies
        ):
            yield np.linalg.eigvalsh(delay_matrix), eigvals

    def time_delay_pdf(self, times: NDArray[np.float64]) -> NDArray[np.float64]:
        global_mean_spacing = 2 * self.ensemble.spectral_radius / self.ensemble.dimension
        j_1_1 = cast(float, jn_zeros(1, 1)[0])
        heisenberg_time = 2 * j_1_1 / global_mean_spacing

        return time_delay_pdf(
            times,
            num_channels=self.num_channels,
            heisenberg_time=heisenberg_time,
        )

    def _add_width_matrix_to_hamiltonian(self, hamiltonian: HermitianMatrix) -> None:
        diag_indices = np.diag_indices(self.num_channels)
        hamiltonian[diag_indices] -= 0.5j * (self.couplings**2)

    def _rotate_coupling_matrix_by_eigvecs(
        self,
        eigvecs: OrthogonalMatrix | UnitaryMatrix,
    ) -> tuple[
        NDArray[np.floating] | NDArray[np.complexfloating],
        NDArray[np.floating] | NDArray[np.complexfloating],
    ]:
        rotated_coupling_matrix = eigvecs[:, : self.num_channels].copy()
        rotated_coupling_matrix *= self.couplings[None, :]

        if np.isrealobj(rotated_coupling_matrix):
            rotated_coupling_matrix_conj = rotated_coupling_matrix
        else:
            rotated_coupling_matrix_conj = np.conjugate(rotated_coupling_matrix)

        return rotated_coupling_matrix, rotated_coupling_matrix_conj

    def _pick_linalg_dtype(self, use_complex_dtype: bool = False) -> np.dtype:
        if use_complex_dtype or self.ensemble.dyson_index != 1:
            return self.ensemble.complex_dtype
        else:
            return self.ensemble.real_dtype

    def _pick_lapack_geev(self, *, use_complex_dtype: bool = False):
        matrix_dtype = self._pick_linalg_dtype(use_complex_dtype=use_complex_dtype)
        return get_lapack_funcs("geev", dtype=matrix_dtype)
