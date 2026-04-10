from __future__ import annotations

from collections.abc import Iterator
from math import comb, factorial

import numpy as np
from attrs import field, frozen
from attrs.validators import ge, le
from numba import njit

from ._many_body import ManyBodyEnsemble
from ..utils import create_block_slice, create_q_body_majorana_terms


@njit(cache=True, fastmath=True)
def _create_syk_matrix_with_complex_coeffs(
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


@njit(cache=True, fastmath=True)
def _create_syk_matrix_with_real_coeffs(
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


@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class SachdevYeKitaevEnsemble(ManyBodyEnsemble):
    parity: int = field(default=0, repr=False)
    q: int = field(
        converter=int,
        validator=[ge(2), le(10), lambda _, __, value: value % 2 == 0],
    )

    @q.validator
    def _q_validator(self, _, value):
        if value > 10:
            raise ValueError("SYK q-parameter cannot be greater than 10.")
        q_limits: dict[int, int] = {2: 32, 4: 32, 6: 26, 8: 24, 10: 22}
        max_q_allowed: int | None = q_limits.get(value)
        if max_q_allowed is not None and self.num_majoranas > max_q_allowed:
            raise ValueError(
                f"SYK with q={value} is supported for num_majoranas <= {max_q_allowed}."
            )

    dyson_index: int = field(init=False, repr=False)

    @dyson_index.default
    def _dyson_index_default(self) -> int:
        map: dict[tuple[int, int], int] = {(0, 0): 1, (0, 4): 4}
        return map.get((self.q % 4, self.num_majoranas % 8), 2) if self.q > 2 else 0

    suppression: float = field(init=False, repr=False)

    @suppression.default
    def _suppression_default(self) -> float:
        if self.q > self.num_majoranas:
            raise ValueError(
                "SYK q-parameter cannot be greater than the number of Majorana particles."
            )
        comb_product_sum: float = np.sum(
            (-1) ** (self.q - k)
            * (comb(self.q, k) * comb(self.num_majoranas - self.q, self.q - k))
            for k in range(self.q + 1)
        )
        return comb_product_sum / comb(self.num_majoranas, self.q)

    std_dev: float = field(init=False, repr=False)

    @std_dev.default
    def _std_dev_default(self) -> float:
        return self.interaction_strength * np.sqrt(
            factorial(self.q - 1) / self.num_majoranas ** (self.q - 1)
        )

    ground_state_energy: float = field(init=False, repr=False)

    @ground_state_energy.default
    def _ground_state_energy_default(self) -> float:
        return (2 * self.std_dev) * np.sqrt(
            comb(self.num_majoranas, self.q) / (1 - self.suppression)
        )

    _q_body_terms_idxs: np.ndarray = field(init=False, repr=False)
    _q_body_terms_data: np.ndarray = field(init=False, repr=False)
    _parity_block: slice = field(init=False, repr=False)

    @_parity_block.default
    def _parity_block_default(self) -> slice:
        return create_block_slice(num_majoranas=self.num_majoranas, parity=self.parity)

    _nickname: str = field(init=False, default="SYK", repr=False)

    def __attrs_post_init__(self) -> None:
        q_body_terms_idxs, q_body_terms_data = create_q_body_majorana_terms(
            q=self.q,
            parity_block=self._parity_block,
            num_majoranas=self.num_majoranas,
            in_real_basis=self.dyson_index == 1,
        )
        object.__setattr__(self, "_q_body_terms_idxs", q_body_terms_idxs)
        object.__setattr__(self, "_q_body_terms_data", q_body_terms_data)

    @property
    def _dir_name(self) -> str:
        return super()._dir_name + f"_{self.q}"

    @property
    def _latex_name(self) -> str:
        return super()._latex_name + f"_{self.q}"

    def generate_matrix(self, use_complex_dtype: bool = True) -> np.ndarray:
        complex_dtype: type[np.complexfloating] = self.complex_dtype.type
        real_dtype: type[np.floating] = self.real_dtype.type
        rng: np.random.Generator = self.rng
        q: int = self.q
        dyson_index: int = self.dyson_index
        size: int = self.dimension
        std_dev: float = self.std_dev
        q_body_idxs: np.ndarray = self._q_body_terms_idxs
        q_body_data: np.ndarray = self._q_body_terms_data

        if dyson_index == 1 and not use_complex_dtype:
            matrix: np.ndarray = np.empty((size, size), real_dtype, order="F")
        else:
            matrix: np.ndarray = np.empty((size, size), complex_dtype, order="F")

        if (q // 2) % 2 == 0:
            _create_syk_matrix_with_real_coeffs(
                matrix, rng, real_dtype, std_dev, q_body_idxs, q_body_data
            )
        else:
            _create_syk_matrix_with_complex_coeffs(
                matrix, rng, real_dtype, std_dev, q_body_idxs, q_body_data
            )
        return matrix

    def matrix_stream(
        self, realizs: int, use_complex_dtype: bool = True
    ) -> Iterator[np.ndarray]:
        complex_dtype: type[np.complexfloating] = self.complex_dtype.type
        real_dtype: type[np.floating] = self.real_dtype.type
        rng: np.random.Generator = self.rng
        q: int = self.q
        dyson_index: int = self.dyson_index
        size: int = self.dimension
        std_dev: float = self.std_dev
        q_body_idxs: np.ndarray = self._q_body_terms_idxs
        q_body_data: np.ndarray = self._q_body_terms_data

        if dyson_index == 1 and not use_complex_dtype:
            matrix: np.ndarray = np.empty((size, size), real_dtype, order="F")
        else:
            matrix: np.ndarray = np.empty((size, size), complex_dtype, order="F")

        if (q // 2) % 2 == 0:
            for _ in range(realizs):
                _create_syk_matrix_with_real_coeffs(
                    matrix, rng, real_dtype, std_dev, q_body_idxs, q_body_data
                )
                yield matrix
        else:
            for _ in range(realizs):
                _create_syk_matrix_with_complex_coeffs(
                    matrix, rng, real_dtype, std_dev, q_body_idxs, q_body_data
                )
                yield matrix

    def spectral_pdf(self, eigvals: np.ndarray, num_terms: int = 100) -> np.ndarray:
        real_dtype: type[np.floating] = self.real_dtype.type
        num_majoranas: int = self.num_majoranas
        q: int = self.q
        eta: float = self.suppression
        std_dev: float = self.std_dev
        e0: float = self.ground_state_energy

        mask: np.ndarray = np.abs(eigvals) < e0
        eigval_sq: np.ndarray = (eigvals[mask] ** 2)[:, None]

        k: np.ndarray = np.arange(num_terms)
        etak1: np.ndarray = eta ** (k + 1)
        term1: np.ndarray = 1 - (4 * eigval_sq / e0**2) * etak1 / (1.0 + etak1) ** 2
        term2: np.ndarray = (1.0 - eta ** (2 * k + 2)) / (1.0 - eta ** (2 * k + 1))

        logP: np.ndarray = (
            np.sum(np.log(term1) + np.log(term2)[None, :], axis=1)
            + np.log(1.0 - eta) / 2
        )

        pdf_vals: np.ndarray = np.exp(logP) * np.sqrt(1.0 - (eigvals[mask] / e0) ** 2)
        pdf_vals /= np.pi * np.sqrt(comb(num_majoranas, q)) * std_dev

        pdf: np.ndarray = np.zeros(eigvals.shape, dtype=real_dtype)
        pdf[mask] = pdf_vals
        return pdf
