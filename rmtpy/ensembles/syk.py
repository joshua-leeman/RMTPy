from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from typing import Any, ClassVar

import attrs
import numba
import numpy as np

import rmtpy.fermions
import rmtpy.polynomials

from .many_body import ManyBodyEnsemble

INITIALISM: str = "SYK"

NUM_MAJORANAS_LIMIT_BY_Q: dict[int, int] = {2: 32, 4: 32, 6: 26, 8: 24, 10: 22}


def compute_dyson_index(syk: SachdevYeKitaevEnsemble) -> int:
    if syk.q == 2:
        return 0

    return {(0, 0): 1, (0, 4): 4}.get((syk.q % 4, syk.num_majoranas % 8), 2)


def compute_standard_deviation(syk: SachdevYeKitaevEnsemble) -> float:
    return syk.interaction_strength * np.sqrt(
        math.factorial(syk.q - 1) / syk.num_majoranas ** (syk.q - 1)
    )


def compute_spectral_radius(syk: SachdevYeKitaevEnsemble) -> float:
    return (2 * syk.std_dev) * np.sqrt(
        math.comb(syk.num_majoranas, syk.q) / (1 - syk.suppression)
    )


def compute_suppression_factor(syk: SachdevYeKitaevEnsemble) -> float:
    return sum(
        ((-1) ** (syk.q - k) / math.comb(syk.num_majoranas, syk.q))
        * (math.comb(syk.q, k) * math.comb(syk.num_majoranas - syk.q, syk.q - k))
        for k in range(syk.q + 1)
    )


def create_spectral_polynomials(
    syk: SachdevYeKitaevEnsemble,
) -> Callable[[np.ndarray, int], np.ndarray]:
    def syk_spectral_polynomials(x: np.ndarray, *, degree: int) -> np.ndarray:
        return rmtpy.polynomials.q_hermite_polynomials(
            x,
            eta=syk.suppression,
            degree=degree,
        )

    return syk_spectral_polynomials


def create_spectral_weight(
    syk: SachdevYeKitaevEnsemble,
) -> Callable[[np.ndarray], np.ndarray]:
    def syk_spectral_weight(energies: np.ndarray) -> np.ndarray:
        return rmtpy.polynomials.q_hermite_polynomial_weight_pdf(
            energies, radius=syk.spectral_radius, eta=syk.suppression
        )

    return syk_spectral_weight


def instantiate_majorana_fermion_basis(
    syk: SachdevYeKitaevEnsemble,
) -> rmtpy.fermions.MajoranaFermionBasis:
    return rmtpy.fermions.MajoranaFermionBasis(
        num_majoranas=syk.num_majoranas,
        in_real_basis=syk.dyson_index == 1,
        is_even_parity=syk.is_even_parity,
    )


@numba.njit(boundscheck=False, cache=True, fastmath=True)
def create_syk_matrix_with_imaginary_prefactor(
    matrix: np.ndarray,
    rng: np.random.Generator,
    real_dtype: type[np.floating[Any]],
    std_dev: float,
    monomials_idxs: np.ndarray,
    monomials_data: np.ndarray,
) -> None:
    num_terms = monomials_data.shape[0]
    coeffs = std_dev * rng.standard_normal(num_terms, real_dtype)

    matrix.fill(0.0)
    for i, monomial_data in enumerate(monomials_data):
        for j, nonzero_monomial_entry in enumerate(monomial_data):
            entry_idx = (monomials_idxs[i, 0, j], monomials_idxs[i, 1, j])
            matrix[entry_idx] += 1j * coeffs[i] * nonzero_monomial_entry


@numba.njit(boundscheck=False, cache=True, fastmath=True)
def create_syk_matrix_without_imaginary_prefactor(
    matrix: np.ndarray,
    rng: np.random.Generator,
    real_dtype: type[np.floating[Any]],
    std_dev: float,
    monomials_idxs: np.ndarray,
    monomials_data: np.ndarray,
) -> None:
    num_terms = monomials_data.shape[0]
    coeffs = std_dev * rng.standard_normal(num_terms, real_dtype)

    matrix.fill(0.0)
    for i, monomial_data in enumerate(monomials_data):
        for j, nonzero_monomial_entry in enumerate(monomial_data):
            entry_idx = (monomials_idxs[i, 0, j], monomials_idxs[i, 1, j])
            matrix[entry_idx] += coeffs[i] * nonzero_monomial_entry


def is_num_majoranas_within_limit(syk: SachdevYeKitaevEnsemble, _, q: int) -> None:
    if syk.num_majoranas > NUM_MAJORANAS_LIMIT_BY_Q[q]:
        raise ValueError(
            f"For the SYK q={q} model, `num_majoranas` cannot exceed "
            f"{NUM_MAJORANAS_LIMIT_BY_Q[q]} due to memory constraints."
        )
    elif q >= syk.num_majoranas:
        raise ValueError(
            f"The SYK q-parameter {q} must be less than the number of majorana "
            f"fermions, here {syk.num_majoranas}."
        )


def normalize_q(q: Any, syk: SachdevYeKitaevEnsemble) -> int:
    q = int(q)
    if q not in NUM_MAJORANAS_LIMIT_BY_Q:
        raise ValueError(
            f"`q` must be one of {tuple(NUM_MAJORANAS_LIMIT_BY_Q)}, got {q}."
        )

    is_num_majoranas_within_limit(syk, None, q)
    return q


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class SachdevYeKitaevEnsemble(ManyBodyEnsemble):
    initialism: ClassVar[str] = INITIALISM

    q: int = attrs.field(
        converter=attrs.Converter(normalize_q, takes_self=True),
    )
    is_even_parity: bool = attrs.field(
        default=True,
        converter=attrs.converters.to_bool,
    )

    suppression: float = attrs.field(
        default=attrs.Factory(compute_suppression_factor, takes_self=True),
        init=False,
        repr=False,
    )
    std_dev: float = attrs.field(
        default=attrs.Factory(compute_standard_deviation, takes_self=True),
        init=False,
        repr=False,
    )
    spectral_radius: float = attrs.field(
        default=attrs.Factory(compute_spectral_radius, takes_self=True),
        init=False,
        repr=False,
    )
    dyson_index: int = attrs.field(
        default=attrs.Factory(compute_dyson_index, takes_self=True),
        init=False,
        repr=False,
    )

    spectral_polynomials: Callable[[np.ndarray], np.ndarray] = attrs.field(
        default=attrs.Factory(create_spectral_polynomials, takes_self=True),
        init=False,
        repr=False,
    )
    spectral_weight: Callable[[np.ndarray], np.ndarray] = attrs.field(
        default=attrs.Factory(create_spectral_weight, takes_self=True),
        init=False,
        repr=False,
    )

    majorana_fermion_basis: rmtpy.fermions.MajoranaFermionBasis = attrs.field(
        default=attrs.Factory(instantiate_majorana_fermion_basis, takes_self=True),
        init=False,
        repr=False,
    )

    _decomposed_q_monomials: rmtpy.fermions.DecomposedSparseArray | None = attrs.field(
        default=None,
        init=False,
        repr=False,
    )

    @property
    def latex_name(self) -> str:
        parity = "even" if self.is_even_parity else "odd"
        return (
            rf"{{\text{{{type(self).initialism}}}}}_{{q = {self.q}}}"
            rf"^\text{{{parity}}}"
            rf"(N_\text{{m}} = {{{self.num_majoranas}}})"
        )

    @property
    def token_name(self) -> str:
        parity = "even" if self.is_even_parity else "odd"
        return f"{super().token_name}_{self.q}_{parity}"

    @property
    def decomposed_q_monomials(self) -> rmtpy.fermions.DecomposedSparseArray:
        if self._decomposed_q_monomials is None:
            object.__setattr__(
                self,
                "_decomposed_q_monomials",
                self.majorana_fermion_basis.create_decomposed_q_monomials(q=self.q),
            )

        return self._decomposed_q_monomials

    def generate_matrix(self, *, use_complex_dtype: bool = False) -> np.ndarray:
        create_syk_matrix = self._pick_syk_matrix_builder()

        matrix = self._empty_matrix(use_complex_dtype=use_complex_dtype)
        create_syk_matrix(
            matrix,
            self.rng,
            self.real_dtype.type,
            self.std_dev,
            self.decomposed_q_monomials[0],
            self.decomposed_q_monomials[1],
        )
        return matrix

    def matrix_stream(
        self, realizs: int, *, use_complex_dtype: bool = False
    ) -> Iterator[np.ndarray]:
        create_syk_matrix = self._pick_syk_matrix_builder()

        matrix = self._empty_matrix(use_complex_dtype=use_complex_dtype)
        for _ in range(realizs):
            create_syk_matrix(
                matrix,
                self.rng,
                self.real_dtype.type,
                self.std_dev,
                self.decomposed_q_monomials[0],
                self.decomposed_q_monomials[1],
            )
            yield matrix

    def _empty_matrix(self, *, use_complex_dtype: bool = False) -> np.ndarray:
        if use_complex_dtype or self.dyson_index != 1:
            return np.empty(
                (self.dimension, self.dimension), self.complex_dtype.type, order="F"
            )
        else:
            return np.empty(
                (self.dimension, self.dimension), self.real_dtype.type, order="F"
            )

    def _pick_syk_matrix_builder(self) -> Callable[..., None]:
        if self.q % 4 == 2:
            return create_syk_matrix_with_imaginary_prefactor
        else:
            return create_syk_matrix_without_imaginary_prefactor
