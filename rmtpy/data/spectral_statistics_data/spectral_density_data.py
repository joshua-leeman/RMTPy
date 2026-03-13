# rmtpy/simulations/spectral_statistics_data/spectral_density_data.py

# Postponed evaluation of annotations
from __future__ import annotations

# Third-party imports
import numpy as np
from attrs import frozen, field

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

    # Count of realizations performed
    realizs_count: int = field(init=False, repr=False)

    # Bins for spectral histogram
    bins: np.ndarray = field(init=False, repr=False)

    # Counts for spectral histogram
    counts: np.ndarray = field(init=False, repr=False)

    # Spectral histogram
    hist: np.ndarray = field(init=False, repr=False)

    # Bins for unfolded spectral histogram
    unf_bins: np.ndarray = field(init=False, repr=False)

    # Counts for unfolded spectral histogram
    unf_counts: np.ndarray = field(init=False, repr=False)

    # Unfolded spectral histogram
    unf_hist: np.ndarray = field(init=False, repr=False)

    @realizs_count.default
    def __default_realizs_count(self) -> int:
        """Initialize count of realizations."""

        # Return zero-initialized count of realizations
        return np.zeros((1,), dtype=int)

    @bins.default
    def __default_bins(self) -> np.ndarray:
        """Generate bins for spectral histogram."""

        # Calculate and return bin edges
        return np.linspace(self.Ei, self.Ef, self.num_bins + 1)

    @counts.default
    def __default_counts(self) -> np.ndarray:
        """Initialize spectral histogram counts."""

        # Return zero-initialized counts
        return np.zeros(self.num_bins)

    @hist.default
    def __default_hist(self) -> np.ndarray:
        """Initialize spectral histogram."""

        # Return empty histogram
        return np.empty(self.num_bins)

    @unf_bins.default
    def __default_unf_bins(self) -> np.ndarray:
        """Generate bins for unfolded spectral histogram."""

        # Calculate and return bin edges
        return np.linspace(self.Ei, self.Ef, self.num_bins + 1)

    @unf_counts.default
    def __default_unf_counts(self) -> np.ndarray:
        """Initialize unfolded spectral histogram counts."""

        # Return zero-initialized counts
        return np.zeros(self.num_bins)

    @unf_hist.default
    def __default_unf_hist(self) -> np.ndarray:
        """Initialize unfolded spectral histogram."""

        # Return empty histogram
        return np.empty(self.num_bins)

    @property
    def realizs(self) -> int:
        """Get the count of realizations performed."""

        # Return realizations from realizs_count array
        return self.realizs_count[0]

    def add_histogram_contribution(self, levels: np.ndarray, unfolded: bool) -> None:
        """Compute the contribution of a spectrum to the spectral histogram."""

        # Select appropriate bins and histogram counts
        bins = self.unf_bins if unfolded else self.bins
        counts = self.unf_counts if unfolded else self.counts

        # Calculate and update histogram counts
        counts[:] += np.histogram(levels, bins=bins)[0]

        # Add realization to count of realizations
        self.realizs_count[0] += 1

    def compute_histograms(self) -> None:
        """Calculate the spectral histograms."""

        # Normalize spectral histogram counts to obtain histogram
        self.hist[:] = self.counts / np.sum(self.counts * np.diff(self.bins))

        # Normalize unfolded spectral histogram counts to obtain histogram
        self.unf_hist[:] = self.unf_counts / np.sum(
            self.unf_counts * np.diff(self.unf_bins)
        )
