from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator

import numpy as np
from attrs import field, frozen
from attrs.validators import ge, gt, le
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.linalg.blas import scopy, dcopy, ccopy, zcopy, sgemm, dgemm, cgemm, zgemm
from scipy.linalg.lapack import sgeev, dgeev, cgeev, zgeev, ssyev, dsyev, cheev, zheev
from scipy.ndimage import gaussian_filter1d
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
    def _default_dimension(self) -> int:
        return 2 ** (self.num_majoranas // 2 - 1)

    ground_state_energy: float = field(init=False, repr=False)

    @ground_state_energy.default
    def _default_ground_state_energy(self) -> float:
        return self.num_majoranas * self.interaction_strength

    _nickname: str = field(init=False, default="MBE", repr=False)

    _numerical_spectral_pdf: PchipInterpolator | None = field(
        init=False, default=None, repr=False
    )
    _numerical_spectral_cdf: PchipInterpolator | None = field(
        init=False, default=None, repr=False
    )

    @property
    def universality_class(self) -> str | None:
        dyson_indices: dict[float, str] = {0: "Poisson", 1: "GOE", 2: "GUE", 4: "GSE"}
        return dyson_indices.get(self.dyson_index, None)

    @property
    def eigval_degeneracy(self) -> int:
        return 2 if self.universality_class == "GSE" else 1

    def eigsys_stream(
        self, realizs: int, use_complex_dtype: bool = False
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        lapack_heev: type = self._pick_lapack_heev(use_complex_dtype)
        for matrix in self.matrix_stream(realizs, use_complex_dtype):
            eigvals, eigvecs, _ = lapack_heev(matrix, compute_v=1, overwrite_a=True)
            yield eigvals, eigvecs

    def eigvals_stream(
        self, realizs: int, use_complex_dtype: bool = False
    ) -> Iterator[np.ndarray]:
        lapack_heev: type = self._pick_lapack_heev(use_complex_dtype)
        for matrix in self.matrix_stream(realizs, use_complex_dtype):
            eigvals = lapack_heev(matrix, compute_v=0, overwrite_a=True)[0]
            yield eigvals

    def spectral_pdf(
        self,
        eigvals: int | float | np.ndarray,
        _factor: float = 1.2,
        _num_bins: int = 200,
        _sigma: float = 2.0,
    ) -> np.ndarray:
        real_dtype: type[np.floating] = self.real_dtype.type
        if isinstance(eigvals, (int, float)):
            eigvals: np.ndarray = np.array([eigvals], dtype=real_dtype)

        if self._numerical_spectral_pdf is None:
            object.__setattr__(
                self,
                "_numerical_spectral_pdf",
                self._create_numerical_spectral_pdf(_num_bins, _factor, _sigma),
            )

        return self._numerical_spectral_pdf(eigvals)

    def spectral_cdf(
        self,
        eigvals: int | float | np.ndarray,
        _factor: float = 1.2,
        _num_bins: int = 200,
        _sigma: float = 2.0,
    ) -> np.ndarray:
        real_dtype: type[np.floating] = self.real_dtype.type
        if isinstance(eigvals, (int, float)):
            eigvals: np.ndarray = np.array([eigvals], dtype=real_dtype)

        if self._numerical_spectral_cdf is None:
            object.__setattr__(
                self,
                "_numerical_spectral_cdf",
                self._create_numerical_spectral_cdf(_num_bins, _factor, _sigma),
            )

        return self._numerical_spectral_cdf(eigvals)

    def unfold(self, eigvals: np.ndarray) -> np.ndarray:
        return self.dimension * (
            self.spectral_cdf(eigvals) - self.spectral_cdf(np.array([0.0]))
        )

    def unfold_widths(self, eigvals: np.ndarray, widths: np.ndarray) -> np.ndarray:
        return self.dimension * (
            self.spectral_cdf(eigvals + widths / 2)
            - self.spectral_cdf(eigvals - widths / 2)
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

    def _create_numerical_spectral_pdf(
        self, num_bins: int = 200, factor: float = 1.2, sigma: float = 2.0
    ) -> None:
        total_counts_per_dimension: int = 2**13 // self.dimension
        realizs: int = max(total_counts_per_dimension, 1)

        energy_0: float = self.ground_state_energy
        bins: np.ndarray = factor * np.linspace(-energy_0, energy_0, num_bins + 1)
        counts: np.ndarray = np.zeros(num_bins)

        for tmp_eigvals in self.eigvals_stream(realizs):
            counts[:] += np.histogram(tmp_eigvals, bins=bins)[0]

        histogram: np.ndarray = counts / np.sum(counts * np.diff(bins))
        smoothed_histogram: np.ndarray = gaussian_filter1d(histogram, sigma=sigma)

        centers: np.ndarray = (bins[:-1] + bins[1:]) / 2
        return PchipInterpolator(centers, smoothed_histogram, extrapolate=True)

    def _create_numerical_spectral_cdf(
        self, num_bins: int = 200, factor: float = 1.2, sigma: float = 2.0
    ) -> None:
        energy_0: float = self.ground_state_energy
        energies: np.ndarray = factor * np.linspace(-energy_0, energy_0, num_bins + 1)
        pdf_values: np.ndarray = self.spectral_pdf(energies, num_bins, factor, sigma)
        cdf_values: np.ndarray = cumulative_trapezoid(pdf_values, energies, initial=0)

        return PchipInterpolator(energies, cdf_values, extrapolate=True)

    def _pick_blas_copy(self, use_complex_dtype: bool) -> type:
        if use_complex_dtype or self.dyson_index != 1:
            if self.complex_dtype == np.complex64:
                return ccopy
            else:
                return zcopy
        else:
            if self.real_dtype == np.float32:
                return scopy
            else:
                return dcopy

    def _pick_blas_gemm(self, use_complex_dtype: bool) -> type:
        if use_complex_dtype or self.dyson_index != 1:
            if self.complex_dtype == np.complex64:
                return cgemm
            else:
                return zgemm
        else:
            if self.real_dtype == np.float32:
                return sgemm
            else:
                return dgemm

    def _pick_lapack_geev(self, use_complex_dtype: bool) -> type:
        if use_complex_dtype or self.dyson_index != 1:
            if self.complex_dtype == np.complex64:
                return cgeev
            else:
                return zgeev
        else:
            if self.real_dtype == np.float32:
                return sgeev
            else:
                return dgeev

    def _pick_lapack_heev(self, use_complex_dtype: bool) -> type:
        if use_complex_dtype or self.dyson_index != 1:
            if self.complex_dtype == np.complex64:
                return cheev
            else:
                return zheev
        else:
            if self.real_dtype == np.float32:
                return ssyev
            else:
                return dsyev

    @abstractmethod
    def generate_matrix(self, use_complex_dtype: bool = False) -> np.ndarray:
        pass

    @abstractmethod
    def matrix_stream(
        self, realizs: int, use_complex_dtype: bool = False
    ) -> Iterator[np.ndarray]:
        pass
