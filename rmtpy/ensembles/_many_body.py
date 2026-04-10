from __future__ import annotations

from functools import lru_cache
from collections.abc import Iterator

import numpy as np
from attrs import field, frozen
from attrs.validators import ge, gt, le
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.linalg import eigh, eigvalsh
from scipy.special import gamma

from ._ensemble import RandomMatrixEnsemble


@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class ManyBodyEnsemble(RandomMatrixEnsemble):
    num_majoranas: int = field(
        converter=int,
        validator=[ge(4), le(32), lambda _, __, value: value % 2 == 0],
        metadata={"dir_name": "Nm", "latex_name": r"N_\textrm{\tiny m}"},
    )
    interaction_strength: float = field(
        default=1.0,
        converter=float,
        validator=gt(0),
        metadata={"dir_name": "J"},
    )

    dyson_index: int | float = field(init=False, default=0, validator=ge(0), repr=False)
    dimension: int = field(init=False)

    @dimension.default
    def _dimension_default(self) -> int:
        return 2 ** (self.num_majoranas // 2 - 1)

    ground_state_energy: float = field(init=False, repr=False)

    @ground_state_energy.default
    def _ground_state_energy_default(self) -> float:
        return self.num_majoranas * self.interaction_strength

    _nickname: str = field(init=False, default="MBE", repr=False)

    @property
    def universality_class(self) -> str | None:
        dyson_indices: dict[float, str] = {0: "Poisson", 1: "GOE", 2: "GUE", 4: "GSE"}
        return dyson_indices.get(self.dyson_index, None)

    @property
    def eigval_degeneracy(self) -> int:
        return 2 if self.universality_class == "GSE" else 1

    def eigsys_stream(self, realizs: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for matrix in self.matrix_stream(realizs):
            yield eigh(matrix, overwrite_a=True, check_finite=False)

    def eigvals_stream(self, realizs: int) -> Iterator[np.ndarray]:
        for matrix in self.matrix_stream(realizs):
            yield eigvalsh(matrix, overwrite_a=True, check_finite=False)

    def generate_matrix(self) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement generate_matrix method.")

    def matrix_stream(self, realizs: int) -> Iterator[np.ndarray]:
        raise NotImplementedError("Subclasses must implement matrix_stream method.")

    def spectral_pdf(self, eigvals: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement spectral_pdf method.")
        # # Define function to compute PDF
        # @lru_cache(maxsize=1)
        # def numerical_pdf(realizs: int = 100, factor: float = 1.1) -> interp1d:
        #     """Create numerical PDF using eigenvalue realizations."""
        #     pass
        # # Return PDF values for given eigenvalues
        # return numerical_pdf()(eigvals)
        # Raise NotImplementedError if not implemented
        # raise NotImplementedError("Subclasses must implement spectral_pdf method.")

    def spectral_cdf(self, eigvals: np.ndarray) -> np.ndarray:
        @lru_cache(maxsize=1)
        def numerical_cdf(num_pts: int = 2**12, factor: int = 1.1) -> interp1d:
            energy_0: float = self.ground_state_energy

            energies: np.ndarray = factor * np.linspace(-energy_0, energy_0, num_pts)
            pdf_vals: np.ndarray = self.spectral_pdf(energies)
            cdf_vals: np.ndarray = cumulative_trapezoid(pdf_vals, energies, initial=0)
            cdf_interp: interp1d = interp1d(
                energies, cdf_vals, bounds_error=False, fill_value=(0, 1)
            )
            return cdf_interp

        return numerical_cdf()(eigvals)

    def unfold(self, eigvals: np.ndarray) -> np.ndarray:
        return self.dimension * (
            self.spectral_cdf(eigvals) - self.spectral_cdf(np.array([0.0]))
        )

    def wigner_surmise(self, spacings: np.ndarray) -> np.ndarray:
        degeneracy: int = self.eigval_degeneracy
        spacings /= degeneracy

        idx: int | float = self.dyson_index
        if idx == 0:
            return np.exp(-spacings)

        a: float = gamma((idx + 2) / 2) ** (idx + 1) / gamma((idx + 1) / 2) ** (idx + 2)
        b: float = (gamma((idx + 2) / 2) / gamma((idx + 1) / 2)) ** 2
        return 2 * a * spacings**idx * np.exp(-b * spacings**2) / degeneracy

    def universal_csff(self, unfolded_times: np.ndarray) -> np.ndarray:
        real_dtype: type[np.floating] = self.real_dtype.type
        dyson_index: int | float = self.dyson_index
        dim: int = self.dimension
        tau: np.ndarray = unfolded_times / (2 * np.pi)

        if dyson_index == 1:
            csff: np.ndarray = np.empty_like(tau, real_dtype)

            m: np.ndarray = tau <= 1
            csff[m] = tau[m] * (2 - np.log(2 * tau[m] + 1)) / dim

            m = tau > 1
            csff[m] = (2 - tau[m] * np.log((2 * tau[m] + 1) / (2 * tau[m] - 1))) / dim

            return csff

        if dyson_index == 2:
            return np.where(tau <= 1, tau / dim, 1 / dim)

        if dyson_index == 4:
            csff: np.ndarray = np.full_like(tau, 2 / dim, real_dtype)
            csff[2 * tau == 1] = np.nan

            m: np.ndarray = (tau < 1) & (2 * tau != 1)
            log_term: np.ndarray = np.log(np.abs(2 * tau[m] - 1))
            csff[m] = 2 * (tau[m] - tau[m] / 2 * log_term) / dim
            return csff

        return np.full_like(tau, 1 / dim, real_dtype)
