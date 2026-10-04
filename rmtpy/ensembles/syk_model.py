import math
from collections.abc import Callable, Iterator
from functools import cached_property
from typing import ClassVar, cast, override

import attrs
import numba
import numpy as np

from ..conversion import RMT_CONVERTER, SourceDict
from ..fermions import (
    MajoranaFermionBasis,
    QMonomialsDataComplex,
    QMonomialsDataReal,
    QMonomialsDecomposed,
    QMonomialsIndices,
)
from ..polynomials import (
    FloatFunction,
    OrthogonalPolynomials,
    q_hermite_polynomial_weight,
    q_hermite_polynomials,
)
from .base_ensemble import RandomMatrixEnsemble
from .many_body_ensemble import HermitianMatrix, ManyBodyEnsemble, RealSymmetricMatrix

INITIALISM: str = "SYK"

NUM_MAJORANAS_LIMIT_BY_Q: dict[int, int] = {2: 32, 4: 32, 6: 26, 8: 24, 10: 22}


@numba.njit(boundscheck=False, cache=True, fastmath=True)
def build_syk_matrix_with_imaginary_prefactor(
    matrix: HermitianMatrix,
    real_dtype: type[np.floating],
    std_dev: float,
    monomials_idxs: QMonomialsIndices,
    monomials_data: QMonomialsDataComplex,
    rng: np.random.Generator,
) -> None:
    num_monomials = monomials_data.shape[0]
    num_nonzeros = monomials_data.shape[1]
    coefficients = std_dev * rng.standard_normal(num_monomials, real_dtype)

    matrix.fill(0.0)
    for i in range(num_monomials):
        for j in range(num_nonzeros):
            entry_idx = (monomials_idxs[i, 0, j], monomials_idxs[i, 1, j])
            matrix[entry_idx] += 1j * coefficients[i] * monomials_data[i, j]


@numba.njit(boundscheck=False, cache=True, fastmath=True)
def build_syk_matrix_without_imaginary_prefactor(
    matrix: RealSymmetricMatrix | HermitianMatrix,
    real_dtype: type[np.floating],
    std_dev: float,
    monomials_idxs: QMonomialsIndices,
    monomials_data: QMonomialsDataReal | QMonomialsDataComplex,
    rng: np.random.Generator,
) -> None:
    num_monomials = monomials_data.shape[0]
    num_nonzeros = monomials_data.shape[1]
    coefficients = std_dev * rng.standard_normal(num_monomials, real_dtype)

    matrix.fill(0.0)
    for i in range(num_monomials):
        for j in range(num_nonzeros):
            entry_idx = (monomials_idxs[i, 0, j], monomials_idxs[i, 1, j])
            matrix[entry_idx] += coefficients[i] * monomials_data[i, j]


def _validate_q(num_majoranas: int, q: int) -> None:
    if q not in NUM_MAJORANAS_LIMIT_BY_Q:
        raise ValueError(
            f"`q` must be one of {tuple(NUM_MAJORANAS_LIMIT_BY_Q)}, got {q}."
        )
    if num_majoranas > NUM_MAJORANAS_LIMIT_BY_Q[q]:
        raise ValueError(
            f"For the SYK q={q} model, `num_majoranas` cannot exceed "
            + f"{NUM_MAJORANAS_LIMIT_BY_Q[q]} due to memory constraints."
        )
    elif q >= num_majoranas:
        raise ValueError(
            f"The SYK q-parameter {q} must be less than the number of majorana "
            + f"fermions, here {num_majoranas}."
        )


def _are_num_majoranas_within_limit(
    syk: SachdevYeKitaevEnsemble,
    _: attrs.Attribute[int],
    q: int,
) -> None:
    _validate_q(syk.num_majoranas, q)


def _compute_dyson_index(syk: SachdevYeKitaevEnsemble) -> int:
    if syk.q == 2:
        return 0

    return {(0, 0): 1, (0, 4): 4}.get((syk.q % 4, syk.num_majoranas % 8), 2)


def _compute_standard_deviation(syk: SachdevYeKitaevEnsemble) -> float:
    conventional_denominator = cast(int, pow(syk.num_majoranas, (syk.q - 1)))
    variance_numerical_factor = math.factorial(syk.q - 1) / conventional_denominator
    return syk.interaction_strength * cast(float, np.sqrt(variance_numerical_factor))


def _compute_suppression_factor(syk: SachdevYeKitaevEnsemble) -> float:
    _validate_q(syk.num_majoranas, syk.q)
    return sum(
        ((-1) ** (syk.q - k) / math.comb(syk.num_majoranas, syk.q))
        * (math.comb(syk.q, k) * math.comb(syk.num_majoranas - syk.q, syk.q - k))
        for k in range(syk.q + 1)
    )


def _compute_spectral_radius(syk: SachdevYeKitaevEnsemble) -> float:
    radius_numerical_factor = math.comb(syk.num_majoranas, syk.q) / (1 - syk.suppression)
    return (2 * syk.std_dev) * cast(float, np.sqrt(radius_numerical_factor))


def _create_majorana_fermion_basis(syk: SachdevYeKitaevEnsemble) -> MajoranaFermionBasis:
    return MajoranaFermionBasis(
        num_majoranas=syk.num_majoranas,
        in_real_basis=syk.dyson_index == 1,
        is_even_parity=syk.is_even_parity,
    )


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class SachdevYeKitaevEnsemble(ManyBodyEnsemble):
    initialism: ClassVar[str] = INITIALISM

    q: int = attrs.field(
        converter=int,
        validator=_are_num_majoranas_within_limit,
    )
    is_even_parity: bool = attrs.field(
        default=True,
        converter=attrs.converters.to_bool,
    )

    suppression: float = attrs.field(
        default=attrs.Factory(_compute_suppression_factor, takes_self=True),
        init=False,
        repr=False,
    )
    std_dev: float = attrs.field(
        default=attrs.Factory(_compute_standard_deviation, takes_self=True),
        init=False,
        repr=False,
    )
    spectral_radius: float = attrs.field(
        default=attrs.Factory(_compute_spectral_radius, takes_self=True),
        init=False,
        repr=False,
    )
    dyson_index: int = attrs.field(
        default=attrs.Factory(_compute_dyson_index, takes_self=True),
        init=False,
        repr=False,
    )

    majorana_fermion_basis: MajoranaFermionBasis = attrs.field(
        default=attrs.Factory(_create_majorana_fermion_basis, takes_self=True),
        init=False,
        repr=False,
    )

    @property
    @override
    def latex_name(self) -> str:
        parity = "even" if self.is_even_parity else "odd"
        return (
            rf"{{\text{{{type(self).initialism}}}}}_{{q = {self.q}}}"
            rf"^\text{{{parity}}}"
            rf"(N_\text{{m}} = {{{self.num_majoranas}}})"
        )

    @property
    @override
    def token_name(self) -> str:
        parity = "even" if self.is_even_parity else "odd"
        return f"{super().token_name}_{self.q}_{parity}"

    @cached_property
    def _decomposed_q_monomials(self) -> QMonomialsDecomposed:
        return self.majorana_fermion_basis.build_decomposed_q_monomials(q=self.q)

    @classmethod
    @override
    def create(
        cls,
        src: SourceDict | RandomMatrixEnsemble,
    ) -> SachdevYeKitaevEnsemble:
        return RMT_CONVERTER.structure(src, cls)

    @override
    def generate_matrix(
        self,
        *,
        use_complex_dtype: bool = False,
    ) -> RealSymmetricMatrix | HermitianMatrix:
        if use_complex_dtype or self.dyson_index != 1:
            matrix = self._allocate_complex_hermitian_matrix_memory()
        else:
            matrix = self._allocate_empty_real_symmetric_matrix_memory()

        build_syk_matrix = self._pick_syk_matrix_builder()
        build_syk_matrix(
            matrix,
            self.real_dtype.type,
            self.std_dev,
            self._decomposed_q_monomials[0],
            self._decomposed_q_monomials[1],
            self.rng,
        )
        return matrix

    @override
    def matrix_stream(
        self,
        realizs: int,
        *,
        use_complex_dtype: bool = False,
    ) -> Iterator[RealSymmetricMatrix | HermitianMatrix]:
        if use_complex_dtype or self.dyson_index != 1:
            matrix = self._allocate_complex_hermitian_matrix_memory()
        else:
            matrix = self._allocate_empty_real_symmetric_matrix_memory()

        build_syk_matrix = self._pick_syk_matrix_builder()
        for _ in range(realizs):
            build_syk_matrix(
                matrix,
                self.real_dtype.type,
                self.std_dev,
                self._decomposed_q_monomials[0],
                self._decomposed_q_monomials[1],
                self.rng,
            )
            yield matrix

    @override
    def assign_spectral_polynomials(self) -> OrthogonalPolynomials:
        def syk_model_spectral_polynomials(
            x: np.ndarray[tuple[int], np.dtype[np.floating]],
            /,
            *,
            degree: int,
        ) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
            return q_hermite_polynomials(x, eta=self.suppression, degree=degree)

        return syk_model_spectral_polynomials

    @override
    def assign_spectral_weight(self) -> FloatFunction:
        def syk_model_spectral_weight(
            energies: np.ndarray[tuple[int], np.dtype[np.floating]],
            /,
        ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
            return q_hermite_polynomial_weight(
                energies,
                support_radius=self.spectral_radius,
                eta=self.suppression,
            )

        return syk_model_spectral_weight

    def _pick_syk_matrix_builder(self) -> Callable[..., None]:
        if self.q % 4 == 2:
            return build_syk_matrix_with_imaginary_prefactor
        else:
            return build_syk_matrix_without_imaginary_prefactor
