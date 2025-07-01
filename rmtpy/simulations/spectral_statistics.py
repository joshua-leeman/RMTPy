# rmtpy/simulations/spectral_statistics.py

# Postponed evaluation of annotations
from __future__ import annotations

# Standard library imports
import os
import shutil
from argparse import ArgumentParser
from attrs import frozen, field

# Third-party imports
import numpy as np
from scipy.special import jn_zeros

# Local imports
from ._simulation import Data, Simulation


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

    # Remove near-duplicate spacings
    spacings = spacings[::degen]

    # Duplicate spacings with degeneracy
    spacings = np.repeat(spacings, degen)

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
@frozen(kw_only=True, repr=False, eq=False, unsafe_hash=True)
class SpectralStatisticsData(Data):
    # Spectral histogram simulation parameters
    spec_num_bins: int
