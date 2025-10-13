# rmtpy/simulations/old_spectral_statistics.py

# Postponed evaluation of annotations
from __future__ import annotations

# Standard library imports
import os
import inspect
import shutil
from argparse import ArgumentParser
from attrs import frozen, field
from attrs.validators import instance_of
from pathlib import Path
from typing import Any

# Third-party imports
import numpy as np
from scipy.special import jn_zeros

# Local imports
from ..ensembles import ManyBodyEnsemble
from .base.simulation import Data, Simulation


# --------------------
# Simulation Functions
# --------------------
def spectral_histogram(levels: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Compute the spectral histogram of an eigenvalue sample."""
    # Calculate histogram counts
    counts, _ = np.histogram(levels, bins=bins)

    # Retrun counts
    return counts


def spacings_histogram(levels: np.ndarray, bins: np.ndarray, degen: int) -> np.ndarray:
    """Compute the nearest-neighbor spacings histogram from an eigenvalue sample."""
    # Compute the nearest-neighbor spacings
    spacings = np.diff(np.sort(levels))

    # Remove near-duplicate spacings and apply degeneracy
    spacings[:] = np.repeat(spacings[::degen], degen)

    # Calculate histogram counts
    counts, _ = np.histogram(spacings, bins=bins)

    # Return counts
    return counts


def sff_moments(levels: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the spectral form factor moments from an eigenvalue sample."""
    # Determine dimension of Hilbert space from levels
    dim = len(levels)

    # Calculate complex exponential terms
    exp_terms = np.exp(-1j * np.outer(levels, times))

    # Calculate contribution to first moment
    mu_1 = np.sum(exp_terms, axis=0) / dim

    # Calculate contribution to second moment
    mu_2 = np.abs(mu_1) ** 2

    # Return first and second moments of spectral form factor
    return mu_1, mu_2


# ------------------------------
# Spectral Statistics Data Class
# ------------------------------
@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class SpectralStatisticsData(Data):
    # Spectral histogram parameters
    spec_num_bins: int = 100
    spec_Ei: float = -1.2  # factor of E0 or D/2
    spec_Ef: float = 1.2  # factor of E0 or D/2

    # Spacings histogram parameters
    spac_num_bins: int = 100
    spac_si: float = 0.0  # factor of global mean spacing
    spac_sf: float = 4.0  # factor of global mean spacing

    # Spectral form factor parameters
    num_times: int = 5000
    logtime_i: float = -0.5  # base = dim
    logtime_f: float = 1.5  # base = dim
    unf_logtime_i: float = -1.5  # base = dim
    unf_logtime_f: float = 0.5  # base = dim

    # Global mean level spacing
    mean_spac: float | None = field(init=False, repr=False, default=None)

    # Bins for spectral histogram
    spec_bins: np.ndarray = field(init=False, repr=False)

    @spec_bins.default
    def __default_spec_bins(self) -> np.ndarray:
        """Generate bins for spectral histogram."""
        # Calculate and return bin edges
        return np.linspace(self.spec_Ei, self.spec_Ef, self.spec_num_bins + 1)

    # Spectral histogram
    spec_hist: np.ndarray = field(init=False, repr=False)

    @spec_hist.default
    def __default_spec_hist(self) -> np.ndarray:
        """Initialize spectral histogram."""
        # Calculate and return zero-initialized histogram
        return np.zeros(self.spec_num_bins)

    # Bins for unfolded spectral histogram
    unf_spec_bins: np.ndarray = field(init=False, repr=False)

    @unf_spec_bins.default
    def __default_unf_spec_bins(self) -> np.ndarray:
        """Generate bins for unfolded spectral histogram."""
        # Calculate and return bin edges
        return np.linspace(self.spec_Ei, self.spec_Ef, self.spec_num_bins + 1)

    # Unfolded spectral histogram
    unf_spec_hist: np.ndarray = field(init=False, repr=False)

    @unf_spec_hist.default
    def __default_unf_spec_hist(self) -> np.ndarray:
        """Initialize unfolded spectral histogram."""
        # Calculate and return zero-initialized histogram
        return np.zeros(self.spec_num_bins)

    # Bins for spacings histogram
    spac_bins: np.ndarray = field(init=False, repr=False)

    @spac_bins.default
    def __default_spac_bins(self) -> np.ndarray:
        """Generate bins for spacings histogram."""
        # Calculate and return bin edges
        return np.linspace(self.spac_si, self.spac_sf, self.spac_num_bins + 1)

    # Nearest-neighbor spacings histogram
    spac_hist: np.ndarray = field(init=False, repr=False)

    @spac_hist.default
    def __default_spac_hist(self) -> np.ndarray:
        """Initialize nearest-neighbor spacings histogram."""
        # Calculate and return zero-initialized histogram
        return np.zeros(self.spac_num_bins)

    # Bins for unfolded spacings histogram
    unf_spac_bins: np.ndarray = field(init=False, repr=False)

    @unf_spac_bins.default
    def __default_unf_spac_bins(self) -> np.ndarray:
        """Generate bins for unfolded spacings histogram."""
        # Calculate and return bin edges
        return np.linspace(self.spac_si, self.spac_sf, self.spac_num_bins + 1)

    # Unfolded nearest-neighbor spacings histogram
    unf_spac_hist: np.ndarray = field(init=False, repr=False)

    @unf_spac_hist.default
    def __default_unf_spac_hist(self) -> np.ndarray:
        """Initialize unfolded nearest-neighbor spacings histogram."""
        # Calculate and return zero-initialized histogram
        return np.zeros(self.spac_num_bins)

    # Times for spectral form factors
    time_array: np.ndarray = field(init=False, repr=False)

    @time_array.default
    def __default_time_array(self) -> np.ndarray:
        """Generate times for spectral form factors."""
        # Calculate and return logarithmic time range
        return np.logspace(self.logtime_i, self.logtime_f, self.num_times)

    # First moment of spectral form factor
    mu_1: np.ndarray = field(init=False, repr=False)

    @mu_1.default
    def __default_mu_1(self) -> np.ndarray:
        """Initialize first moment of spectral form factor."""
        # Calculate and return zero-initialized first moment
        return np.zeros(self.num_times, dtype=np.complex128)

    # Second moment of spectral form factor
    mu_2: np.ndarray = field(init=False, repr=False)

    @mu_2.default
    def __default_mu_2(self) -> np.ndarray:
        """Initialize second moment of spectral form factor."""
        # Calculate and return zero-initialized second moment
        return np.zeros(self.num_times, dtype=np.float64)

    # Times for unfolded spectral form factors
    unf_time_array: np.ndarray = field(init=False, repr=False)

    @unf_time_array.default
    def __default_unf_time_array(self) -> np.ndarray:
        """Generate times for unfolded spectral form factors."""
        # Calculate and return logarithmic time range
        return np.logspace(self.unf_logtime_i, self.unf_logtime_f, self.num_times)

    # First moment of unfolded spectral form factor
    unf_mu_1: np.ndarray = field(init=False, repr=False)

    @unf_mu_1.default
    def __default_unf_mu_1(self) -> np.ndarray:
        """Initialize first moment of unfolded spectral form factor."""
        # Calculate and return zero-initialized first moment
        return np.zeros(self.num_times, dtype=np.complex128)

    # Second moment of unfolded spectral form factor
    unf_mu_2: np.ndarray = field(init=False, repr=False)

    @unf_mu_2.default
    def __default_unf_mu_2(self) -> np.ndarray:
        """Initialize second moment of unfolded spectral form factor."""
        # Calculate and return zero-initialized second moment
        return np.zeros(self.num_times, dtype=np.float64)


# ------------------------------------
# Spectral Statistics Simulation Class
# ------------------------------------
@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class SpectralStatistics(Simulation):
    # Simulation data
    data: SpectralStatisticsData = field(
        factory=SpectralStatisticsData, validator=instance_of(SpectralStatisticsData)
    )

    def __attrs_post_init__(self) -> None:
        """Post-initialization hook to validate simulation and normalize data."""
        # Call parent post-init method
        super().__attrs_post_init__()

        # Ensure ensemble is a ManyBodyEnsemble
        if not isinstance(self.ensemble, ManyBodyEnsemble):
            raise TypeError(
                f"SpectralStatistics requires a ManyBodyEnsemble, "
                f"got {type(self.ensemble).__name__} instead."
            )

        # Alias ensemble dimension, energy scale, and simulation data
        dim = self.ensemble.dim
        E0 = self.ensemble.E0
        data = self.data

        # Calculate global mean level spacing and store in data
        object.__setattr__(data, "mean_spac", 2 * E0 / dim)

        # Store first positive zero of 1st Bessel function
        j_1_1 = jn_zeros(1, 1)[0]

        # Scale spectral histogram bins by energy scale
        data.spec_bins[:] *= E0

        # Scale unfolded spectral histogram bins by half dimension
        data.unf_spec_bins[:] *= dim / 2

        # Scale spacings histogram bins by global mean spacing
        data.spac_bins[:] *= data.mean_spac

        # Change base of time_array to dimension and divide by inverse energy scale
        data.time_array[:] **= np.log(dim) / np.log(10.0)
        data.time_array[:] *= j_1_1 / E0

        # Change base of unf_time_array to dimension and multiply by 2π
        data.unf_time_array[:] **= np.log(dim) / np.log(10.0)
        data.unf_time_array[:] *= 2 * np.pi

    def combine_data(self, path: str | Path) -> None:
        # Define function to prepare simulation metadata for comparison
        pass

    def run_part_1(self) -> None:
        """Run the first part of the spectral statistics simulation."""
        # Alias ensemble, eigenvalue degeneracy, and data
        ensemble: ManyBodyEnsemble = self.ensemble
        data: SpectralStatisticsData = self.data
        degen: int = ensemble.degeneracy

        # Loop over spectrum realizations and calculate spectral statistics
        for eigvals in ensemble.eigvals_stream(self.realizs):
            # Calculate spectral histogram counts
            data.spec_hist[:] += spectral_histogram(eigvals, data.spec_bins)

            # Calculate nearest-neighbor spacings histogram counts
            data.spac_hist[:] += spacings_histogram(eigvals, data.spac_bins, degen)

            # Calculate spectral form factor moments and update data
            mu_1, mu_2 = sff_moments(eigvals, data.time_array)
            data.mu_1[:] += mu_1
            data.mu_2[:] += mu_2

            # Unfold eigenvalues
            unf_eigvals = ensemble.unfold(eigvals)

            # Calculate unfolded spectral histogram counts
            data.unf_spec_hist[:] += spectral_histogram(unf_eigvals, data.unf_spec_bins)

            # Calculate unfolded nearest-neighbor spacings histogram counts
            data.unf_spac_hist[:] += spacings_histogram(
                unf_eigvals, data.unf_spac_bins, degen
            )

            # Calculate unfolded spectral form factor moments and update data
            unf_mu_1, unf_mu_2 = sff_moments(unf_eigvals, data.unf_time_array)
            data.unf_mu_1[:] += unf_mu_1
            data.unf_mu_2[:] += unf_mu_2
