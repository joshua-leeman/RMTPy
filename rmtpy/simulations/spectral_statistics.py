# rmtpy.simulations.spectral_statistics.py


# =======================================
# 1. Imports
# =======================================
# Standard library imports
from __future__ import annotations
import os
import shutil
from argparse import ArgumentParser
from dataclasses import dataclass, field

# Third-party imports
import numpy as np
from scipy.special import jn_zeros

# Local application imports
from rmtpy.simulations._mc import MonteCarlo, _parse_mc_args


# =======================================
# 2. Functions
# =======================================
def spectral_histogram(levels: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Calculate the spectral histogram counts of eigenvalue sample."""
    # Calculate histogram counts
    counts, _ = np.histogram(levels, bins=bins)

    # Return counts
    return counts


def spacings_histogram(levels: np.ndarray, degen: int, bins: np.ndarray) -> np.ndarray:
    """Calculate the nearest neighbor level spacing histogram counts."""
    # Calculate nearest neighbor level spacings
    spacings = np.diff(levels)

    # If degeneracy greater than one, clean spacings
    if degen > 1:
        # Remove near-duplicate spacings
        spacings = spacings[1::degen]

        # Duplicate spacings with degeneracy
        spacings = np.repeat(spacings, degen)

    # Calculate histogram counts
    counts, _ = np.histogram(spacings, bins=bins)

    # Return counts and mean nn-level spacing
    return counts


def form_factor_moments(
    levels: np.ndarray, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the spectral form factors at given times."""
    # Determine dimension of Hilbert space from levels
    dim = len(levels)

    # Calculate complex exponential terms
    exp_terms = np.exp(-1j * np.outer(levels, times))

    # Calculate contribution to first moment
    mu_1 = np.sum(exp_terms, axis=0) / dim

    # Calculate contribution to second moment
    mu_2 = np.abs(mu_1) ** 2

    # Return spectral form factors
    return mu_1, mu_2


# =======================================
# 3. Configuration Class
# =======================================
@dataclass(repr=False, eq=False, frozen=True, kw_only=True, slots=True)
class Config:
    # Spectral histogram simulation parameters
    spec_num_bins: int = 100
    spec_i: float = -1.2  # factor of E0 or D/2
    spec_f: float = 1.2  # factor of E0 or D/2

    # NN-level spacing simulation parameters
    spac_num_bins: int = 100
    spac_i: float = 0.0  # factor of global mean spacing
    spac_f: float = 4.0  # factor of global mean spacing

    # SFF simulation parameters
    num_times: int = 5000
    logtime_i: float = -0.5  # base = dim
    logtime_f: float = 1.5  # base = dim
    unf_logtime_i: float = -1.5  # base = dim
    unf_logtime_f: float = 0.5  # base = dim
    resolve_dip: bool = False

    def _create_spec_bins(self) -> np.ndarray:
        """Create spectral histogram bin edges."""
        return np.linspace(self.spec_i, self.spec_f, self.spec_num_bins + 1)

    def _create_spac_bins(self) -> np.ndarray:
        """Create nearest neighbor level spacing bin edges."""
        return np.linspace(self.spac_i, self.spac_f, self.spac_num_bins + 1)

    def _create_times_array(self, base: float) -> np.ndarray:
        """Create times array."""
        # Begin with a equally spaced array
        times_array = np.logspace(
            self.logtime_i, self.logtime_f, self.num_times, base=base, dtype=np.float64
        )

        # Resolve form factor dip if time range covers it
        if self.logtime_i < 0.0 and self.logtime_f > 0.5:
            # Define array with extra time points
            extra_times = np.logspace(
                0.0, 0.5, 2 * self.num_times, base=base, dtype=np.float64
            )

            # Append extra time points to the original array and return it
            times_array = np.sort(np.concatenate((times_array, extra_times)))

            # Change resolve_dip to True
            object.__setattr__(self, "resolve_dip", True)

        # Return times array
        return times_array

    def _create_unf_times_array(self, base: float) -> np.ndarray:
        """Create unfolded times array."""
        # Begin with a equally spaced array
        times_array = np.logspace(
            self.unf_logtime_i,
            self.unf_logtime_f,
            self.num_times,
            base=base,
            dtype=np.float64,
        )

        # Resolve form factor dip if time range covers it
        if self.unf_logtime_i < -1.0 and self.unf_logtime_f > -0.25:
            # Define array with extra time points
            extra_times = np.logspace(
                -1.0, -0.25, 2 * self.num_times, base=base, dtype=np.float64
            )

            # Append extra time points to the original array and return it
            times_array = np.sort(np.concatenate((times_array, extra_times)))

            # Change resolve_dip to True
            object.__setattr__(self, "resolve_dip", True)

        # Return times array
        return times_array


# =======================================
# 4. Simulation Class
# =======================================
@dataclass(repr=False, eq=False, frozen=True, kw_only=True, slots=True)
class SpectralStatistics(MonteCarlo):
    # Simulation configuration
    config: Config = field(default_factory=Config)

    def run(self) -> None:
        """Run the spectral statistics simulation."""
        # Store first positive zero of 1st Bessel function
        j_1_1 = jn_zeros(1, 1)[0]

        # Estimate global mean level spacing
        global_mean_spacing = 2 * self.ensemble.E0 / self.ensemble.dim

        # Spectral histogram bin edges
        spec_bin_edges = self.config._create_spec_bins() * self.ensemble.E0

        # Unfolded spectral histogram bin edges
        unf_spec_bin_edges = self.config._create_spec_bins() * self.ensemble.dim / 2

        # Nearest neighbor level spacing bin edges
        spac_bin_edges = self.config._create_spac_bins() * global_mean_spacing

        # Unfolded nearest neighbor level spacing bin edges
        unf_spac_bin_edges = self.config._create_spac_bins()

        # Create times array
        times = self.config._create_times_array(base=self.ensemble.dim)
        times *= j_1_1 / self.ensemble.E0

        # Create unfolded times array
        unf_times = self.config._create_unf_times_array(base=self.ensemble.dim)
        unf_times *= 2 * np.pi

        # Initialize spectral histogram
        spec_hist = np.zeros(self.config.spec_num_bins)

        # Initialize unfolded spectral histogram
        unf_spec_hist = np.zeros(self.config.spec_num_bins)

        # Initialize nearest neighbor level spacing histogram
        spac_hist = np.zeros(self.config.spac_num_bins)

        # Initialize unfolded nearest neighbor level spacing histogram
        unf_spac_hist = np.zeros(self.config.spac_num_bins)

        # Determine number of time points based on resolve_dip flag
        num_times = (
            3 * self.config.num_times
            if self.config.resolve_dip
            else self.config.num_times
        )

        # Initialize spectral form factor moments
        mu_1 = np.zeros(num_times, dtype=np.complex128)
        mu_2 = np.zeros(num_times, dtype=np.float64)

        # Initialize unfolded spectral form factors
        unf_mu_1 = np.zeros(num_times, dtype=np.complex128)
        unf_mu_2 = np.zeros(num_times, dtype=np.float64)

        # Loop over spectrum realizations and calculate statistics
        for eigvals in self.ensemble.eigvals_stream(self.realizs):
            # Calculate and accumulate spectral histogram
            spec_hist += spectral_histogram(eigvals, spec_bin_edges)

            # Calculate and accumulate nearest neighbor level spacings
            spac_hist += spacings_histogram(
                eigvals, self.ensemble.degen, spac_bin_edges
            )

            # Calculate and accumulate spectral form factors
            moments = form_factor_moments(eigvals, times)
            mu_1 += moments[0]
            mu_2 += moments[1]

            # Unfold spectrum
            eigvals = self.ensemble.unfold(eigvals)

            # Calculate and accumulate unfolded spectral histogram
            unf_spec_hist += spectral_histogram(eigvals, unf_spec_bin_edges)

            # Calculate and accumulate unfolded nearest neighbor level spacings
            unf_spac_hist += spacings_histogram(
                eigvals, self.ensemble.degen, unf_spac_bin_edges
            )

            # Calculate and accumulate unfolded spectral form factors
            unf_moments = form_factor_moments(eigvals, unf_times)
            unf_mu_1 += unf_moments[0]
            unf_mu_2 += unf_moments[1]

        # Determine ID of worker
        process_id = os.environ.get("SLURM_PROCID", "0")

        # Create file name for output
        file_name = os.path.join(self.outdir, f"{process_id}.npz")

        # Create temporary file name for output
        temp_file_name = os.path.join(self.outdir, f"{process_id}.npz.tmp")

        # Scale realizations by number of SLURM tasks
        realizs = int(os.environ.get("SLURM_NTASKS", 1)) * self.realizs

        # Save results to temporary file first
        with open(temp_file_name, "wb") as temp_file:
            # Store compressed data
            np.savez_compressed(
                temp_file,
                realizs=realizs,
                spec_bin_edges=spec_bin_edges,
                spec_hist=spec_hist,
                unf_spec_bin_edges=unf_spec_bin_edges,
                unf_spec_hist=unf_spec_hist,
                spac_bin_edges=spac_bin_edges,
                spac_hist=spac_hist,
                unf_spac_bin_edges=unf_spac_bin_edges,
                unf_spac_hist=unf_spac_hist,
                times=times,
                mu_1=mu_1,
                mu_2=mu_2,
                unf_times=unf_times,
                unf_mu_1=unf_mu_1,
                unf_mu_2=unf_mu_2,
            )

            # Flush file
            temp_file.flush()

            # Force write to disk
            os.fsync(temp_file.fileno())

        # Rename temporary file to final file name
        shutil.move(temp_file_name, file_name)


# =======================================
# 5. Main Function
# =======================================
def main() -> None:
    """Main function to run the spectral statistics simulation."""
    # Create argument parser
    parser = ArgumentParser(description="Spectral Statistics Monte Carlo Simulation")

    # Parse command line arguments
    mc_args = _parse_mc_args(parser)

    # Create an instance of the SpectralStatistics class
    mc = SpectralStatistics(**mc_args)

    # Run simulation
    mc.run()


# If this script is run directly, execute main function
if __name__ == "__main__":
    # Run main function
    main()
