from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from typing import ClassVar

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


def compute_spectral_radius(syk: SachdevYeKitaevEnsemble) -> float:
    return (2 * syk.std_dev) * np.sqrt(
        math.comb(syk.num_majoranas, syk.q) / (1 - syk.suppression)
    )


def compute_standard_deviation(syk: SachdevYeKitaevEnsemble) -> float:
    return syk.interaction_strength * np.sqrt(
        math.factorial(syk.q - 1) / syk.num_majoranas ** (syk.q - 1)
    )


def compute_suppression_factor(syk: SachdevYeKitaevEnsemble) -> float:
    return np.sum(
        ((-1) ** (syk.q - k) / math.comb(syk.num_majoranas, syk.q))
        * (math.comb(syk.q, k) * math.comb(syk.num_majoranas - syk.q, syk.q - k))
        for k in range(syk.q + 1)
    )


def choose_matrix_block_slice(syk: SachdevYeKitaevEnsemble) -> tuple[slice, slice]:
    return rmtpy.fermions.choose_block_slice_from_parity(
        syk.num_majoranas, syk.is_even_parity
    )


def create_spectral_weight(
    syk: SachdevYeKitaevEnsemble,
) -> Callable[[np.ndarray], np.ndarray]:
    def syk_spectral_weight(energies: np.ndarray) -> np.ndarray:
        return rmtpy.polynomials.q_hermite_polynomial_weight_pdf(
            energies, syk.spectral_radius, syk.suppression
        )

    return syk_spectral_weight


def is_num_majoranas_within_limit(syk: SachdevYeKitaevEnsemble, _, q: int) -> None:
    if syk.num_majoranas > NUM_MAJORANAS_LIMIT_BY_Q[q]:
        raise ValueError(
            f"For the SYK q={q} model, `num_majoranas` cannot exceed "
            "{NUM_MAJORANAS_LIMIT_BY_Q[q]} due to memory constraints."
        )


@numba.njit(cache=True, fastmath=True)
def create_syk_matrix_with_complex_entries(
    matrix: np.ndarray,
    rng: np.random.Generator,
    real_dtype: type[np.floating],
    std_dev: float,
    term_idxs: np.ndarray,
    term_data: np.ndarray,
) -> np.ndarray:
    num_terms: int = term_data.shape[0]
    coeffs: np.ndarray = std_dev * rng.standard_normal(num_terms, real_dtype)
    matrix.fill(0.0)
    for term_num in range(num_terms):
        for entry in range(term_data.shape[1]):
            matrix[term_idxs[term_num, 0, entry], term_idxs[term_num, 1, entry]] += (
                1j * coeffs[term_num] * term_data[term_num, entry]
            )


@numba.njit(cache=True, fastmath=True)
def create_syk_matrix_with_real_entries(
    matrix: np.ndarray,
    rng: np.random.Generator,
    real_dtype: type[np.floating],
    std_dev: float,
    term_idxs: np.ndarray,
    term_data: np.ndarray,
) -> np.ndarray:
    num_terms: int = term_data.shape[0]
    coeffs: np.ndarray = std_dev * rng.standard_normal(num_terms, real_dtype)
    matrix.fill(0.0)
    for term_num in range(num_terms):
        for entry in range(term_data.shape[1]):
            matrix[term_idxs[term_num, 0, entry], term_idxs[term_num, 1, entry]] += (
                coeffs[term_num] * term_data[term_num, entry]
            )


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class SachdevYeKitaevEnsemble(ManyBodyEnsemble):
    initialism: ClassVar[str] = INITIALISM

    q: int = attrs.field(
        converter=int,
        validator=[
            attrs.validators.in_(NUM_MAJORANAS_LIMIT_BY_Q),
            is_num_majoranas_within_limit,
        ],
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
        default=rmtpy.polynomials.q_hermite_polynomials,
        init=False,
        repr=False,
    )
    spectral_weight: Callable[[np.ndarray], np.ndarray] = attrs.field(
        default=attrs.Factory(create_spectral_weight, takes_self=True),
        init=False,
        repr=False,
    )

    _parity_block: tuple[slice, slice] = attrs.field(
        default=attrs.Factory(choose_matrix_block_slice, takes_self=True),
        init=False,
        repr=False,
    )
    _q_body_term_decomps: tuple[tuple[np.ndarray, ...], ...] | None = attrs.field(
        default=None,
        init=False,
        repr=False,
    )

    @property
    def path_name(self) -> str:
        return super().latex_name + f"_{self.q}"

    @property
    def token_name(self) -> str:
        return super().token_name + f"_{self.q}"

    @property
    def q_body_term_decomps(self) -> tuple[tuple[np.ndarray, ...], ...]:
        if self._q_body_term_decomps is None:
            q_body_majorana_terms: tuple[tuple[np.ndarray, ...], ...] = (
                rmtpy.fermions.create_q_body_majorana_terms(
                    q=self.q,
                    parity_block=self._parity_block,
                    num_majoranas=self.num_majoranas,
                    in_real_basis=self.dyson_index == 1,
                )
            )
            object.__setattr__(self, "_q_body_term_decomps", q_body_majorana_terms)

        return self._q_body_term_decomps

    def generate_matrix(self, use_complex_dtype: bool = False) -> np.ndarray:
        matrix = self._initialize_matrix(use_complex_dtype)
        create_syk_matrix = self._pick_syk_matrix_builder(use_complex_dtype)
        create_syk_matrix(
            matrix,
            self.rng,
            self.real_dtype.type,
            self.std_dev,
            self.q_body_term_decomps[0],
            self.q_body_term_decomps[1],
        )
        return matrix

    def matrix_stream(
        self, realizs: int, use_complex_dtype: bool = False
    ) -> Iterator[np.ndarray]:
        matrix = self._initialize_matrix(use_complex_dtype)
        create_syk_matrix = self._pick_syk_matrix_builder(use_complex_dtype)
        for _ in range(realizs):
            create_syk_matrix(
                matrix,
                self.rng,
                self.real_dtype.type,
                self.std_dev,
                self.q_body_term_decomps[0],
                self.q_body_term_decomps[1],
            )
            yield matrix

    def _initialize_matrix(self, use_complex_dtype: bool = False) -> np.ndarray:
        size: int = self.dimension
        if use_complex_dtype or self.dyson_index != 1:
            return np.empty((size, size), self.complex_dtype.type, order="F")
        else:
            return np.empty((size, size), self.real_dtype.type, order="F")

    def _pick_syk_matrix_builder(
        self, use_complex_dtype: bool = False
    ) -> Callable[[np.ndarray], np.ndarray]:
        if use_complex_dtype or self.dyson_index != 1:
            return create_syk_matrix_with_complex_entries
        else:
            return create_syk_matrix_with_real_entries
