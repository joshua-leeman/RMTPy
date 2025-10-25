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


# --------------------------------
# Thouless Time Estimator Function
# --------------------------------
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
        return np.logspace(self.logt_i, self.logt_f, self.num_times, order="F")

    @mu_1.default
    def __default_mu_1(self) -> np.ndarray:
        """Initialize first moment of spectral form factor."""

        # Calculate and return zero-initialized first moment
        return np.zeros(self.num_times, dtype=np.complex128, order="F")

    @mu_2.default
    def __default_mu_2(self) -> np.ndarray:
        """Initialize second moment of spectral form factor."""

        # Calculate and return zero-initialized second moment
        return np.zeros(self.num_times, dtype=np.float64, order="F")

    @sff.default
    def __default_sff(self) -> np.ndarray:
        """Initialize spectral form factor with zeros."""

        # Return zero-initialized spectral form factor
        return np.zeros(self.num_times, dtype=np.float64, order="F")

    @csff.default
    def __default_csff(self) -> np.ndarray:
        """Initialize connected spectral form factor with zeros."""

        # Return zero-initialized connected spectral form factor
        return np.zeros(self.num_times, dtype=np.float64, order="F")

    @unf_times.default
    def __default_unf_times(self) -> np.ndarray:
        """Generate times for unfolded spectral form factors."""

        # Calculate and return logarithmic time range
        return np.logspace(self.unf_logt_i, self.unf_logt_f, self.num_times, order="F")

    @unf_mu_1.default
    def __default_unf_mu_1(self) -> np.ndarray:
        """Initialize first moment of unfolded spectral form factor."""

        # Calculate and return zero-initialized first moment
        return np.zeros(self.num_times, dtype=np.complex128, order="F")

    @unf_mu_2.default
    def __default_unf_mu_2(self) -> np.ndarray:
        """Initialize second moment of unfolded spectral form factor."""

        # Calculate and return zero-initialized second moment
        return np.zeros(self.num_times, dtype=np.float64, order="F")

    @unf_sff.default
    def __default_unf_sff(self) -> np.ndarray:
        """Initialize unfolded spectral form factor with zeros."""

        # Return zero-initialized unfolded spectral form factor
        return np.zeros(self.num_times, dtype=np.float64, order="F")

    @unf_csff.default
    def __default_unf_csff(self) -> np.ndarray:
        """Initialize unfolded connected spectral form factor with zeros."""

        # Return zero-initialized unfolded connected spectral form factor
        return np.zeros(self.num_times, dtype=np.float64, order="F")

    @property
    def thouless_time(self) -> float:
        """Estimate the Thouless time from the spectral form factor data."""

        # Estimate and return the Thouless time
        return thouless_time(self.times, self.sff)
