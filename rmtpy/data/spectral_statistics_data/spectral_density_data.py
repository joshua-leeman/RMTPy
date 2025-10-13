# rmtpy/simulations/spectral_statistics_data/spectral_density_data.py

# Postponed evaluation of annotations
from __future__ import annotations

# Standard library imports
from attrs import frozen, field

# Third-party imports
import numpy as np

# Local application imports
from .._data import Data


# ---------------------------
# Spectral Density Data Class
# ---------------------------
@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class SpectralDensityData(Data):

    # Number of histogram bins
    num_bins: int = 100

    # Range for spectral histogram
    Ei: float = -1.2  # factor of E0 or D/2
    Ef: float = 1.2  # factor of E0 or D/2

    # Bins for spectral histogram
    bins: np.ndarray = field(init=False, repr=False)

    # Spectral histogram
    hist: np.ndarray = field(init=False, repr=False)

    # Bins for unfolded spectral histogram
    unf_bins: np.ndarray = field(init=False, repr=False)

    # Unfolded spectral histogram
    unf_hist: np.ndarray = field(init=False, repr=False)

    @bins.default
    def __default_bins(self) -> np.ndarray:
        """Generate bins for spectral histogram."""

        # Calculate and return bin edges
        return np.linspace(self.Ei, self.Ef, self.num_bins + 1)

    @hist.default
    def __default_hist(self) -> np.ndarray:
        """Initialize spectral histogram."""

        # Calculate and return zero-initialized histogram
        return np.zeros(self.num_bins)

    @unf_bins.default
    def __default_unf_bins(self) -> np.ndarray:
        """Generate bins for unfolded spectral histogram."""

        # Calculate and return bin edges
        return np.linspace(self.Ei, self.Ef, self.num_bins + 1)

    @unf_hist.default
    def __default_unf_hist(self) -> np.ndarray:
        """Initialize unfolded spectral histogram."""

        # Calculate and return zero-initialized histogram
        return np.zeros(self.num_bins)
