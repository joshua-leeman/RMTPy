# rmtpy/simulations/spectral_statistics/form_factors_data.py

# Postponed evaluation of annotations
from __future__ import annotations

# Standard library imports
from attrs import frozen, field

# Third-party imports
import numpy as np

# Local application imports
from ..base import Data


# ----------------------
# Form Factor Data Class
# ----------------------
@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class FormFactorsData(Data):
    """Data class for spectral form factor."""

    # Number of times for spectral form factors
    num_times: int = 5000

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
        return np.logspace(self.logt_i, self.logt_f, self.num_times)

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
        return np.zeros(self.num_times, dtype=np.float64)

    @csff.default
    def __default_csff(self) -> np.ndarray:
        """Initialize connected spectral form factor with zeros."""

        # Return zero-initialized connected spectral form factor
        return np.zeros(self.num_times, dtype=np.float64)

    @unf_times.default
    def __default_unf_times(self) -> np.ndarray:
        """Generate times for unfolded spectral form factors."""

        # Calculate and return logarithmic time range
        return np.logspace(self.unf_logt_i, self.unf_logt_f, self.num_times)

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
        return np.zeros(self.num_times, dtype=np.float64)

    @unf_csff.default
    def __default_unf_csff(self) -> np.ndarray:
        """Initialize unfolded connected spectral form factor with zeros."""

        # Return zero-initialized unfolded connected spectral form factor
        return np.zeros(self.num_times, dtype=np.float64)
