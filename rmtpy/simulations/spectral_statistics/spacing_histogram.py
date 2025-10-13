# rmtpy/simulations/spectral_statistics/spacing_histogram.py

# Postponed evaluation of annotations
from __future__ import annotations

# Standard library imports
from attrs import frozen, field

# Third-party imports
import numpy as np

# Local imports
from ..base.data import Data


# ----------------------------
# Spacing Histogram Data Class
# ----------------------------
@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class SpacingsData(Data):
    """Data class for spacing histogram."""

    # Number of histogram bins
    num_bins: int = 100

    # Range for spacings histogram
    si: float = 0.0  # factor of global mean spacing
    sf: float = 4.0  # factor of global mean spacing

    # Global mean level spacing
    mean: float | None = field(init=False, repr=False, default=None)

    # Bins for spacings histogram
    bins: np.ndarray = field(init=False, repr=False)

    # Nearest-neighbor spacings histogram
    hist: np.ndarray = field(init=False, repr=False)

    # Bins for unfolded spacings histogram
    unf_bins: np.ndarray = field(init=False, repr=False)

    # Unfolded nearest-neighbor spacings histogram
    unf_hist: np.ndarray = field(init=False, repr=False)

    @bins.default
    def __default_bins(self) -> np.ndarray:
        """Generate bins for spacings histogram."""
        # Calculate and return bin edges
        return np.linspace(self.si, self.sf, self.num_bins + 1)

    @hist.default
    def __default_hist(self) -> np.ndarray:
        """Initialize nearest-neighbor spacings histogram."""
        # Calculate and return zero-initialized histogram
        return np.zeros(self.num_bins)

    @unf_bins.default
    def __default_unf_bins(self) -> np.ndarray:
        """Generate bins for unfolded spacings histogram."""
        # Calculate and return bin edges
        return np.linspace(self.si, self.sf, self.num_bins + 1)

    @unf_hist.default
    def __default_unf_hist(self) -> np.ndarray:
        """Initialize unfolded nearest-neighbor spacings histogram."""
        # Calculate and return zero-initialized histogram
        return np.zeros(self.num_bins)
