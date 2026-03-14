# rmtpy/simulations/spectral_statistics_data/form_factors_data.py

# Postponed evaluation of annotations
from __future__ import annotations

# Third-party imports
import numpy as np
from attrs import frozen, field
from attrs.validators import gt
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks

# Local application imports
from .._data import Data


# -----------------------
# Thouless Time Estimator
# -----------------------
def thouless_time(times: np.ndarray, sff: np.ndarray) -> float:
    """Estimate the Thouless time from the spectral form factor data."""

    # Find local maxima of spectral form factor
    max_idx, _ = find_peaks(sff)

    # Interpolate envelope of local maxima using PCHIP
    pchip = PchipInterpolator(times[max_idx], sff[max_idx])

    # Determine index of absolute minimum of interpolated maxima
    start = max_idx[0]
    stop = max_idx[-1] + 1
    env = pchip(times[start:stop])
    rel_idx = np.argmin(env)
    idx = np.searchsorted(times, times[start:stop][rel_idx])

    # Return the Thouless time
    return float(times[idx])


# ----------------------
# Form Factor Data Class
# ----------------------
@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class FormFactorsData(Data):

    # Number of times for spectral form factors
    num_times: int = field(converter=int, validator=gt(0), default=1000)

    # Logarithmic range for times
    logt_i: float = -0.5  # base = dim
    logt_f: float = 1.5  # base = dim

    # Logarithmic range for unfolded times
    unf_logt_i: float = -1.5  # base = dim
    unf_logt_f: float = 0.5  # base = dim

    # Times for spectral form factors
    times: np.ndarray = field(init=False, repr=False)

    # Count of realizations performed
    realizs_count: int = field(init=False, repr=False)

    # First moment of spectral form factor
    mu_1: np.ndarray = field(init=False, repr=False)

    # Second moment of spectral form factor
    mu_2: np.ndarray = field(init=False, repr=False)

    # Spectral form factor
    sff: np.ndarray = field(init=False, repr=False)

    # Connected spectral form factor
    csff: np.ndarray = field(init=False, repr=False)

    # Times for unfolded spectral form factors
    unf_times: np.ndarray = field(init=False, repr=False)

    # Count of unfolded realizations performed
    unf_realizs_count: int = field(init=False, repr=False)

    # First moment of unfolded spectral form factor
    unf_mu_1: np.ndarray = field(init=False, repr=False)

    # Second moment of unfolded spectral form factor
    unf_mu_2: np.ndarray = field(init=False, repr=False)

    # Unfolded spectral form factor
    unf_sff: np.ndarray = field(init=False, repr=False)

    # Unfolded connected spectral form factor
    unf_csff: np.ndarray = field(init=False, repr=False)

    @times.default
    def __default_times(self) -> np.ndarray:
        """Generate times for spectral form factors."""

        # Calculate and return logarithmic time range
        return np.logspace(self.logt_i, self.logt_f, self.num_times)

    @realizs_count.default
    def __default_realizs_count(self) -> int:
        """Initialize count of realizations."""

        # Return zero-initialized count of realizations
        return np.zeros((1,), dtype=int)

    @mu_1.default
    def __default_mu_1(self) -> np.ndarray:
        """Initialize first moment of spectral form factor."""

        # Calculate and return zero-initialized first moment
        return np.zeros(self.num_times, dtype=np.complex128)

    @mu_2.default
    def __default_mu_2(self) -> np.ndarray:
        """Initialize second moment of spectral form factor."""

        # Calculate and return zero-initialized second moment
        return np.zeros(self.num_times, dtype=np.float64)

    @sff.default
    def __default_sff(self) -> np.ndarray:
        """Initialize spectral form factor with zeros."""

        # Return zero-initialized spectral form factor
        return np.empty(self.num_times, dtype=np.float64)

    @csff.default
    def __default_csff(self) -> np.ndarray:
        """Initialize connected spectral form factor with zeros."""

        # Return zero-initialized connected spectral form factor
        return np.empty(self.num_times, dtype=np.float64)

    @unf_times.default
    def __default_unf_times(self) -> np.ndarray:
        """Generate times for unfolded spectral form factors."""

        # Calculate and return logarithmic time range
        return np.logspace(self.unf_logt_i, self.unf_logt_f, self.num_times)

    @unf_realizs_count.default
    def __default_unf_realizs_count(self) -> int:
        """Initialize count of unfolded realizations."""

        # Return zero-initialized count of unfolded realizations
        return np.zeros((1,), dtype=int)

    @unf_mu_1.default
    def __default_unf_mu_1(self) -> np.ndarray:
        """Initialize first moment of unfolded spectral form factor."""

        # Calculate and return zero-initialized first moment
        return np.zeros(self.num_times, dtype=np.complex128)

    @unf_mu_2.default
    def __default_unf_mu_2(self) -> np.ndarray:
        """Initialize second moment of unfolded spectral form factor."""

        # Calculate and return zero-initialized second moment
        return np.zeros(self.num_times, dtype=np.float64)

    @unf_sff.default
    def __default_unf_sff(self) -> np.ndarray:
        """Initialize unfolded spectral form factor with zeros."""

        # Return zero-initialized unfolded spectral form factor
        return np.empty(self.num_times, dtype=np.float64)

    @unf_csff.default
    def __default_unf_csff(self) -> np.ndarray:
        """Initialize unfolded connected spectral form factor with zeros."""

        # Return zero-initialized unfolded connected spectral form factor
        return np.empty(self.num_times, dtype=np.float64)

    @property
    def realizs(self) -> int:
        """Get the count of realizations performed."""

        # Return realizations from realizs_count array
        return self.realizs_count[0]

    @property
    def unf_realizs(self) -> int:
        """Get the count of unfolded realizations performed."""

        # Return realizations from unf_realizs_count array
        return self.unf_realizs_count[0]

    @property
    def thouless_time(self) -> float:
        """Estimate the Thouless time from the spectral form factor data."""

        # Estimate and return the Thouless time
        return thouless_time(self.times, self.sff)

    def compute_moment_contributions(self, levels: np.ndarray, unfolded: bool) -> None:
        """Compute the first and second moments of the spectral form factor from an eigenvalue sample."""

        # Select and alias appropriate times, realizations, and moment arrays
        if unfolded:
            times = self.unf_times
            realizs = self.unf_realizs_count
            mu_1 = self.unf_mu_1
            mu_2 = self.unf_mu_2
        else:
            times = self.times
            realizs = self.realizs_count
            mu_1 = self.mu_1
            mu_2 = self.mu_2

        # =============================================================

        # Determine dimension of Hilbert space from levels
        dim = len(levels)

        # Calculate contribution to first moment
        mu_1_contrib = np.sum(np.exp(-1j * np.outer(levels, times)), axis=0) / dim

        # Add contribution to first moment
        mu_1[:] += mu_1_contrib

        # Add contribution to second moment
        mu_2[:] += np.abs(mu_1_contrib) ** 2

        # Add realization to count of realizations
        realizs[0] += 1

    def compute_form_factors(self) -> None:
        """Compute the spectral form factor and connected spectral form factor from the moments."""

        # Alias realizations, moments, and form factor arrays
        realizs = self.realizs
        mu_1 = self.mu_1
        mu_2 = self.mu_2
        sff = self.sff
        csff = self.csff

        # Alias unfolded realizations, moments, and form factor arrays
        unf_realizs = self.unf_realizs
        unf_mu_1 = self.unf_mu_1
        unf_mu_2 = self.unf_mu_2
        unf_sff = self.unf_sff
        unf_csff = self.unf_csff

        # =============================================================

        # Calculate spectral form factor
        sff[:] = mu_2 / realizs

        # Calculate connected spectral form factor
        csff[:] = sff - np.abs(mu_1 / realizs) ** 2

        # Calculate unfolded spectral form factor
        unf_sff[:] = unf_mu_2 / unf_realizs

        # Calculate unfolded connected spectral form factor
        unf_csff[:] = unf_sff - np.abs(unf_mu_1 / unf_realizs) ** 2
