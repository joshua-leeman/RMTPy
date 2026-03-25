# rmtpy/simulations/spectral_statistics_simulation.py

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
from ..dataclasses.spectral_statistics_data import (
    SpectralDensityData,
    SpacingHistogramData,
    FormFactorsData,
)
from ..ensembles import ManyBodyEnsemble
from ..plotting import FormFactorsPlot, SpacingHistogramPlot, SpectralDensityPlot


# ---------------------------------------
# Spectral Statistics Simulation Function
# ---------------------------------------
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

    # Unfolded spectral density plot instance
    unf_spectral_plot: SpectralDensityPlot | None = field(
        init=False, repr=False, default=None
    )

    # Spacings histogram plot instance
    spacing_hist: SpacingHistogramPlot | None = field(
        init=False, repr=False, default=None
    )

    # Unfolded spacings histogram plot instance
    unf_spacing_hist: SpacingHistogramPlot | None = field(
        init=False, repr=False, default=None
    )

    # Spectral form factors plot instance
    form_factors_plot: FormFactorsPlot | None = field(
        init=False, repr=False, default=None
    )

    # Unfolded spectral form factors plot instance
    unf_form_factors_plot: FormFactorsPlot | None = field(
        init=False, repr=False, default=None
    )

    def initialize_plots(self) -> None:
        """Initialize plot instances for spectral statistics simulation."""

        # Initialize spectral plot
        object.__setattr__(
            self, "spectral_plot", SpectralDensityPlot(data=self.spectral_data)
        )

        # Initialize unfolded spectral plot
        object.__setattr__(
            self,
            "unf_spectral_plot",
            SpectralDensityPlot(data=self.spectral_data, unfold=True),
        )

        # Initialize spacing histogram plot
        object.__setattr__(
            self, "spacing_hist", SpacingHistogramPlot(data=self.spacings_data)
        )

        # Initialize unfolded spacing histogram plot
        object.__setattr__(
            self,
            "unf_spacing_hist",
            SpacingHistogramPlot(data=self.spacings_data, unfold=True),
        )

        # Initialize form factors plot
        object.__setattr__(
            self, "form_factors_plot", FormFactorsPlot(data=self.factors_data)
        )

        # Initialize unfolded form factors plot
        object.__setattr__(
            self,
            "unf_form_factors_plot",
            FormFactorsPlot(data=self.factors_data, unfold=True),
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

        # =================================================

        # Check if ensemble is ManyBodyEnsemble
        if not isinstance(ensemble, ManyBodyEnsemble):
            raise TypeError("Ensemble must be an instance of ManyBodyEnsemble.")

        # Store first positive zero of 1st Bessel function
        j_1_1 = jn_zeros(1, 1)[0]

        # Calculate theoretical global mean level spacing and store in data
        object.__setattr__(spacings, "theory_mean", 2 * E0 / dim)

        # Scale spectral histogram bins by energy scale
        spectral.bins[:] *= E0

        # Scale unfolded spectral histogram bins by half dimension
        spectral.unf_bins[:] *= dim / 2

        # Scale spacings histogram bins by theoretical global mean spacing
        spacings.bins[:] *= spacings.theory_mean

        # Change base of factors.times to dimension
        factors.times[:] **= np.log(dim) / np.log(10.0)

        # Scale times by j_1_1 / E0
        factors.times[:] *= j_1_1 / E0

        # Change base of unf_times to dimension
        factors.unf_times[:] **= np.log(dim) / np.log(10.0)

        # Scale unfolded times by 2 * pi
        factors.unf_times[:] *= 2 * np.pi

        # Loop over spectrum realizations and calculate spectral statistics
        for eigvals in ensemble.eigvals_stream(realizs):
            # Calculate and update spectral histogram counts
            spectral.add_histogram_contribution(eigvals, unfolded=False)

            # Calculate and update spacings histogram counts
            spacings.add_histogram_contribution(eigvals, degen, unfolded=False)

            # Calculate and update form factor moments
            factors.compute_moment_contributions(eigvals, unfolded=False)

            # Unfold eigenvalues
            unf_eigvals = ensemble.unfold(eigvals)

            # Calculate and update unfolded spectral histogram counts
            spectral.add_histogram_contribution(unf_eigvals, unfolded=True)

            # Calculate and update unfolded spacings histogram counts
            spacings.add_histogram_contribution(unf_eigvals, degen, unfolded=True)

            # Calculate and update unfolded form factor moments
            factors.compute_moment_contributions(unf_eigvals, unfolded=True)

    def calculate_statistics(self) -> None:
        """Calculate final spectral statistics from Monte Carlo data."""

        # Alias simulation data
        spectral: SpectralDensityData = self.spectral_data
        spacings: SpacingHistogramData = self.spacings_data
        factors: FormFactorsData = self.factors_data

        # =================================================

        # Calculate spectral histograms from counts
        spectral.compute_histograms()

        # Calculate spacings histograms from counts
        spacings.compute_histograms()

        # Calculate spectral form factors from moments
        factors.compute_form_factors()

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
