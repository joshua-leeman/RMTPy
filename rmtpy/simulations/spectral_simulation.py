# rmtpy/simulations/spectral_statistics/spectral_simulation.py

# Postponed evaluation of annotations
from __future__ import annotations

# Standard library imports
from pathlib import Path

# Third-party imports
import numpy as np
from attrs import frozen, field
from scipy.special import jn_zeros

# Local application imports
from ._simulation import Simulation
from ..data.spectral_statistics_data import SpectralDensityData
from ..data.spectral_statistics_data import SpacingHistogramData
from ..data.spectral_statistics_data import FormFactorsData
from ..ensembles import ManyBodyEnsemble
from ..plotting.spectral_statistics_plots import SpectralDensityPlot


# --------------------
# Simulation Functions
# --------------------
def spectral_histogram(levels: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Compute the spectral histogram of an eigenvalue sample."""

    # Calculate histogram counts
    counts, _ = np.histogram(levels, bins=bins)

    # Return counts
    return counts


def spacings_histogram(levels: np.ndarray, bins: np.ndarray, degen: int) -> np.ndarray:
    """Compute the nearest-neighbor spacings histogram from an eigenvalue sample."""

    # Compute the nearest-neighbor spacings
    spacings = np.diff(np.sort(levels))

    # Remove near-duplicate spacings and apply degeneracy
    if degen > 1:
        spacings = np.repeat(spacings[1::degen], degen)

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


# ---------------------------------
# Random Matrix Simulation Function
# ---------------------------------
def spectral_statistics(ensemble: ManyBodyEnsemble, realizs: int) -> None:
    """Perform a spectral statistics simulation for a given ensemble."""

    # Checks if ensemble is ManyBodyEnsemble
    if not isinstance(ensemble, ManyBodyEnsemble):
        raise TypeError("Ensemble must be an instance of ManyBodyEnsemble.")

    # Checks if realizs is a positive integer
    if not isinstance(realizs, int) or realizs <= 0:
        raise ValueError("Number of realizations must be a positive integer.")

    # Create simulation instance
    sim = SpectralStatistics(ensemble=ensemble, realizs=realizs)

    # Run simulation
    sim.run()


# ------------------------------------
# Spectral Statistics Simulation Class
# ------------------------------------
@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class SpectralStatistics(Simulation):

    # Spectral density data
    spectral_data: SpectralDensityData = field(factory=SpectralDensityData, repr=False)

    # Spacings histogram data
    spacings_data: SpacingHistogramData = field(
        factory=SpacingHistogramData, repr=False
    )

    # Spectral form factor data
    factors_data: FormFactorsData = field(factory=FormFactorsData, repr=False)

    # Spectral density plot instance
    spectral_plot: SpectralDensityPlot | None = field(
        init=False, repr=False, default=None
    )

    def __attrs_post_init__(self) -> None:
        """Initialize metadata after object creation."""

        # Call parent post-init method
        super().__attrs_post_init__()

        # Initialize spectral plot
        object.__setattr__(
            self, "spectral_plot", SpectralDensityPlot(data=self.spectral_data)
        )

    def realize_monte_carlo(self) -> None:
        """Realize Monte Carlo sample of spectral statistics."""

        # Alias ensemble
        ensemble: ManyBodyEnsemble = self.ensemble

        # Alias ensemble parameters
        dim: int = ensemble.dim
        E0: float = ensemble.E0
        degen: int = ensemble.degeneracy

        # Alias number of realizations
        realizs = self.realizs

        # Alias simulation data
        spectral: SpectralDensityData = self.spectral_data
        spacings: SpacingHistogramData = self.spacings_data
        factors: FormFactorsData = self.factors_data

        # Check if ensemble is ManyBodyEnsemble
        if not isinstance(ensemble, ManyBodyEnsemble):
            raise TypeError("Ensemble must be an instance of ManyBodyEnsemble.")

        # Store first positive zero of 1st Bessel function
        j_1_1 = jn_zeros(1, 1)[0]

        # Calculate global mean level spacing and store in data
        object.__setattr__(spacings, "mean", 2 * E0 / dim)

        # Scale spectral histogram bins by energy scale
        spectral.bins[:] *= E0

        # Scale unfolded spectral histogram bins by half dimension
        spectral.unf_bins[:] *= dim / 2

        # Scale spacings histogram bins by global mean spacing
        spacings.bins[:] *= spacings.mean

        # Change base of factors.times to dimension and divide by inverse energy scale
        factors.times[:] **= np.log(dim) / np.log(10.0)
        factors.times[:] *= j_1_1 / E0

        # Change base of unf_times to dimension and multiply by 2π
        factors.unf_times[:] **= np.log(dim) / np.log(10.0)
        factors.unf_times[:] *= 2 * np.pi

        # Loop over spectrum realizations and calculate spectral statistics
        for eigvals in ensemble.eigvals_stream(realizs):
            # Calculate spectral histogram counts
            spectral.hist[:] += spectral_histogram(eigvals, spectral.bins)

            # Calculate nearest-neighbor spacings histogram counts
            spacings.hist[:] += spacings_histogram(eigvals, spacings.bins, degen)

            # Calculate spectral form factor moments and update data
            mu_1, mu_2 = sff_moments(eigvals, factors.times)
            factors.mu_1[:] += mu_1
            factors.mu_2[:] += mu_2

            # Unfold eigenvalues
            unfvals = ensemble.unfold(eigvals)

            # Calculate unfolded spectral histogram counts
            spectral.unf_hist[:] += spectral_histogram(unfvals, spectral.unf_bins)

            # Calculate unfolded nearest-neighbor spacings histogram counts
            spacings.unf_hist[:] += spacings_histogram(
                unfvals, spacings.unf_bins, degen
            )

            # Calculate unfolded spectral form factor moments and update data
            unf_mu_1, unf_mu_2 = sff_moments(unfvals, factors.unf_times)
            factors.unf_mu_1[:] += unf_mu_1
            factors.unf_mu_2[:] += unf_mu_2

    def calculate_statistics(self) -> None:
        """Calculate final spectral statistics from Monte Carlo data."""

        # Alias number of realizations
        realizs = self.realizs

        # Alias simulation data
        spectral: SpectralDensityData = self.spectral_data
        spacings: SpacingHistogramData = self.spacings_data
        factors: FormFactorsData = self.factors_data

        # Normalize spectral histogram
        spectral.hist[:] /= np.sum(spectral.hist * np.diff(spectral.bins))

        # Normalize unfolded spectral histogram
        spectral.unf_hist[:] /= np.sum(spectral.unf_hist * np.diff(spectral.unf_bins))

        # Normalize spacings histogram
        spacings.hist[:] /= np.sum(spacings.hist * np.diff(spacings.bins))

        # Normalize unfolded spacings histogram
        spacings.unf_hist[:] /= np.sum(spacings.unf_hist * np.diff(spacings.unf_bins))

        # Calculate first and second moments of spectral form factor
        factors.mu_1[:] /= realizs
        factors.mu_2[:] /= realizs

        # Calculate connected spectral form factor
        factors.csff[:] = factors.sff - np.abs(factors.mu_1) ** 2

        # Calculate unfolded first and second moments of spectral form factor
        factors.unf_mu_1[:] /= realizs
        factors.unf_mu_2[:] /= realizs

        # Calculate unfolded connected spectral form factor
        factors.unf_csff[:] = factors.unf_sff - np.abs(factors.unf_mu_1) ** 2

    def run(self, out_dir: str | Path = "output") -> None:
        """Run the spectral statistics simulation."""

        # Realize Monte Carlo sample of spectral statistics
        self.realize_monte_carlo()

        # Calculate final spectral statistics from Monte Carlo data
        self.calculate_statistics()

        # Ensure path is a Path object
        out_dir = Path(out_dir)

        # Alias base directory path
        base_dir = out_dir / self.to_dir

        # Create base directory if it does not exist
        base_dir.mkdir(parents=True, exist_ok=True)

        # Save simulation data to output directory
        self.save_data(out_dir=base_dir)

        # Generate and save plots to output directory
        self.save_plots(out_dir=base_dir)
