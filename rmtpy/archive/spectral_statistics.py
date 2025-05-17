# rmtpy.simulations.spectral_statistics.py


# =======================================
# 1. Imports
# =======================================
# Standard library imports
from __future__ import annotations
from argparse import ArgumentParser
from dataclasses import asdict, dataclass, field
from multiprocessing import Pool, set_start_method
from time import time
from typing import Any

# Third-party imports
import numpy as np
from scipy.special import jn_zeros

# Local application imports
from rmtpy.plotting.spectral_statistics.spectral_histogram import SpectralHistogram
from rmtpy.plotting.spectral_statistics.spacings_histogram import SpacingsHistogram
from rmtpy.plotting.spectral_statistics.form_factors_plot import FormFactorPlot
from rmtpy.archive._mc import MonteCarlo, _parse_mc_args
from rmtpy.utils import get_ensemble, configure_matplotlib


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


def _worker_func(args: dict[str, Any]) -> dict[np.ndarray]:
    """Worker function for parallel processing."""
    # Unpack ensemble arguments
    ens_args = args["ens_args"]

    # Find and initialize ensemble
    ensemble = get_ensemble(ens_args)

    # Unpack number of realizations
    realizs = args["realizs"]

    # Initialize configuration
    config = Config(**args["config"])

    # Spectral histogram bin edges
    spec_bin_edges = config._create_spec_bins() * ensemble.E0

    # Initialize spectral histogram
    spec_hist = np.zeros(config.spec_num_bins)

    # Unfolded spectral histogram bin edges
    unf_spec_bin_edges = config._create_spec_bins() * ensemble.dim / 2

    # Initialize unfolded spectral histogram
    unf_spec_hist = np.zeros(config.spec_num_bins)

    # Estimate global mean level spacing
    global_mean_spacing = 2 * ensemble.E0 / ensemble.dim

    # Nearest neighbor level spacing bin edges
    spac_bin_edges = config._create_spac_bins() * global_mean_spacing

    # Initialize nearest neighbor level spacing histogram
    spac_hist = np.zeros(config.spac_num_bins)

    # Unfolded nearest neighbor level spacing bin edges
    unf_spac_bin_edges = config._create_spac_bins()

    # Initialize unfolded nearest neighbor level spacing histogram
    unf_spac_hist = np.zeros(config.spac_num_bins)

    # Store first positive zero of 1st Bessel function
    j_1_1 = jn_zeros(1, 1)[0]

    # Create times array
    times = config._create_times_array(base=ensemble.dim)
    times *= j_1_1 / ensemble.E0

    # Initialize spectral form factor moments
    mu_1 = np.zeros(config.num_times, dtype=np.complex128)
    mu_2 = np.zeros(config.num_times, dtype=np.float64)

    # Create unfolded times array
    unf_times = config._create_unf_times_array(base=ensemble.dim)
    unf_times *= 2 * np.pi

    # Initialize unfolded spectral form factors
    unf_mu_1 = np.zeros(config.num_times, dtype=np.complex128)
    unf_mu_2 = np.zeros(config.num_times, dtype=np.float64)

    # Loop over spectrum realizations and calculate statistics
    for eigvals in ensemble.eigvals_stream(realizs):
        # Calculate and accumulate spectral histogram
        spec_hist += spectral_histogram(eigvals, spec_bin_edges)

        # Calculate and accumulate nearest neighbor level spacings
        spac_hist += spacings_histogram(eigvals, ensemble.degen, spac_bin_edges)

        # Calculate and accumulate spectral form factors
        moments = form_factor_moments(eigvals, times)
        mu_1 += moments[0]
        mu_2 += moments[1]

        # Unfold spectrum
        eigvals = ensemble.unfold(eigvals)

        # Calculate and accumulate unfolded spectral histogram
        unf_spec_hist += spectral_histogram(eigvals, unf_spec_bin_edges)

        # Calculate and accumulate unfolded nearest neighbor level spacings
        unf_spac_hist += spacings_histogram(eigvals, ensemble.degen, unf_spac_bin_edges)

        # Calculate and accumulate unfolded spectral form factors
        unf_moments = form_factor_moments(eigvals, unf_times)
        unf_mu_1 += unf_moments[0]
        unf_mu_2 += unf_moments[1]

    # Return results
    return {
        "spec_hist": spec_hist,
        "unf_spec_hist": unf_spec_hist,
        "spac_hist": spac_hist,
        "unf_spac_hist": unf_spac_hist,
        "mu_1": mu_1,
        "mu_2": mu_2,
        "unf_mu_1": unf_mu_1,
        "unf_mu_2": unf_mu_2,
    }


# =======================================
# 3. Configuration Dataclass
# =======================================
@dataclass(repr=False, eq=False, kw_only=True, slots=True)
class Config:
    # Spectral histogram simulation parameters
    spec_num_bins: int = 50
    spec_i: float = -1.2  # factor of E0 or D/2
    spec_f: float = 1.2  # factor of E0 or D/2
    spec_filename: str = "spectrum"

    # NN-level spacing simulation parameters
    spac_num_bins: int = 50
    spac_i: float = 0.0  # factor of global mean
    spac_f: float = 4.0  # factor of global mean
    spac_filename: str = "spacings"

    # SFF simulation parameters
    num_times: int = 5000
    logtime_i: float = -0.5  # base = dim
    logtime_f: float = 1.5  # base = dim
    unf_logtime_i: float = -1.5  # base = dim
    unf_logtime_f: float = 0.5  # base = dim
    sff_filename: str = "form_factors"

    def _create_spec_bins(self) -> np.ndarray:
        """Create spectral histogram bin edges."""
        return np.linspace(self.spec_i, self.spec_f, self.spec_num_bins + 1)

    def _create_spac_bins(self) -> np.ndarray:
        """Create nearest neighbor level spacing bin edges."""
        return np.linspace(self.spac_i, self.spac_f, self.spac_num_bins + 1)

    def _create_times_array(self, base: float) -> np.ndarray:
        """Create times array."""
        return np.logspace(
            self.logtime_i, self.logtime_f, self.num_times, base=base, dtype=np.float64
        )

    def _create_unf_times_array(self, base: float) -> np.ndarray:
        """Create unfolded times array."""
        return np.logspace(
            self.unf_logtime_i,
            self.unf_logtime_f,
            self.num_times,
            base=base,
            dtype=np.float64,
        )


# =======================================
# 4. Simulation Class
# =======================================
@dataclass(repr=False, eq=False, kw_only=True, slots=True)
class SpectralStatistics(MonteCarlo):
    # Configuration
    config: Config = field(default_factory=Config)

    def run(self) -> None:
        """Run the spectral statistics simulation."""
        # Start timer
        start_time = time()

        # Create arguments for workers
        worker_args = self._create_worker_args()

        # Run workers in parallel
        with Pool(processes=self.workers) as pool:
            results = pool.map(_worker_func, worker_args)  # type: ignore

        # Process spectral histograms
        self._process_spectral_histograms(results)

        # Process nearest neighbor level spacings
        self._process_spacing_histograms(results)

        # Process spectral form factors
        self._process_form_factors(results)

        # End timer and print elapsed time
        elapsed_time = time() - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

    def _create_worker_args(self) -> list[dict[str, Any]]:
        """Create arguments for workers."""
        # Calculate realizations per worker and remainder
        realizs_per_worker, remainder = divmod(self.realizs, self.workers)

        # Initialize realizations list
        realizs_array = np.full(self.workers, realizs_per_worker)

        # Distribute remainder realizations
        realizs_array[:remainder] += 1

        # Create dictionary representation of ensemble
        ens_args = self.ensemble._to_dict_str()

        # Create list of worker arguments as list of dictionaries
        worker_args = [
            {
                "ens_args": ens_args,
                "realizs": realizs_array[i],
                "config": asdict(self.config),
            }
            for i in range(self.workers)
        ]

        # Return list of worker arguments
        return worker_args

    def _process_spectral_histograms(self, results: dict) -> None:
        """Process spectral histograms."""
        # Spectral histogram bin edges
        bins = self.config._create_spec_bins() * self.E0

        # Unfolded spectral histogram bin edges
        unf_bins = self.config._create_spec_bins() * self.dim / 2

        # Combine spectral histogram
        spec_hist = np.sum([res["spec_hist"] for res in results], axis=0)

        # Combine unfolded spectral histogram
        unf_spec_hist = np.sum([res["unf_spec_hist"] for res in results], axis=0)

        # Normalize histograms
        spec_hist /= np.sum(spec_hist * np.diff(bins))
        unf_spec_hist /= np.sum(unf_spec_hist * np.diff(unf_bins))

        # Store spectral histogram data file name
        path = f"{self.output_dir}/{self.config.spec_filename}.npz"

        # Store unfolded spectral histogram data file name
        unf_path = f"{self.output_dir}/{self.config.spec_filename}_unfolded.npz"

        # Save spectral histogram to compressed file
        np.savez_compressed(path, bin_edges=bins, counts=spec_hist)

        # Save unfolded spectral histogram to compressed file
        np.savez_compressed(unf_path, bin_edges=unf_bins, counts=unf_spec_hist)

        # Initialize spectral histogram
        plot = SpectralHistogram(data_path=path, unfold=False)

        # Plot spectral histogram
        plot.plot()

        # Initialize unfolded spectral histogram
        unf_plot = SpectralHistogram(data_path=unf_path, unfold=True)

        # Plot unfolded spectral histogram
        unf_plot.plot()

    def _process_spacing_histograms(self, results: dict) -> None:
        """Process nearest neighbor level spacing histograms."""
        # Estimate global mean level spacing
        global_mean_spacing = 2 * self.E0 / self.dim

        # Nearest neighbor level spacing bin edges
        bins = self.config._create_spac_bins() * global_mean_spacing

        # Unfolded nearest neighbor level spacing bin edges
        unf_bins = self.config._create_spac_bins()

        # Combine nearest neighbor level spacing histogram
        spac_hist = np.sum([res["spac_hist"] for res in results], axis=0)

        # Combine unfolded nearest neighbor level spacing histogram
        unf_spac_hist = np.sum([res["unf_spac_hist"] for res in results], axis=0)

        # Normalize histograms
        spac_hist /= np.sum(spac_hist * np.diff(bins))
        unf_spac_hist /= np.sum(unf_spac_hist * np.diff(unf_bins))

        # Store spectral histogram data file name
        path = f"{self.output_dir}/{self.config.spac_filename}.npz"

        # Store unfolded spectral histogram data file name
        unf_path = f"{self.output_dir}/{self.config.spac_filename}_unfolded.npz"

        # Save nearest neighbor level spacing histogram to compressed file
        np.savez_compressed(path, bin_edges=bins, counts=spac_hist)

        # Save unfolded nearest neighbor level spacing histogram to compressed file
        np.savez_compressed(unf_path, bin_edges=unf_bins, counts=unf_spac_hist)

        # Initialize spectral histogram
        plot = SpacingsHistogram(data_path=path, unfold=False)

        # Plot spectral histogram
        plot.plot()

        # Initialize unfolded spectral histogram
        unf_plot = SpacingsHistogram(data_path=unf_path, unfold=True)

        # Plot unfolded spectral histogram
        unf_plot.plot()

    def _process_form_factors(self, results: dict) -> None:
        """Process spectral form factors."""
        # Store first positive zero of 1st Bessel function
        j_1_1 = jn_zeros(1, 1)[0]

        # Recreate times array
        times = self.config._create_times_array(base=self.dim)
        times *= j_1_1 / self.E0

        # Recreate unfolded times array
        unf_times = self.config._create_unf_times_array(base=self.dim)
        unf_times *= 2 * np.pi

        # Combine spectral form factor moments
        mu_1 = np.sum([res["mu_1"] for res in results], axis=0)
        mu_2 = np.sum([res["mu_2"] for res in results], axis=0)

        # Combine unfolded spectral form factor moments
        unf_mu_1 = np.sum([res["unf_mu_1"] for res in results], axis=0)
        unf_mu_2 = np.sum([res["unf_mu_2"] for res in results], axis=0)

        # Calculate spectral form factors
        sff = mu_2 / self.realizs
        csff = sff - np.abs(mu_1) ** 2 / self.realizs**2

        # Calculate unfolded spectral form factors
        unf_sff = unf_mu_2 / self.realizs
        unf_csff = unf_sff - np.abs(unf_mu_1) ** 2 / self.realizs**2

        # Store spectral histogram data file name
        path = f"{self.output_dir}/{self.config.sff_filename}.npz"

        # Store unfolded spectral histogram data file name
        unf_path = f"{self.output_dir}/{self.config.sff_filename}_unfolded.npz"

        # Save spectral form factors to compressed file
        np.savez_compressed(path, times=times, sff=sff, csff=csff)

        # Save unfolded spectral form factors to compressed file
        np.savez_compressed(unf_path, times=unf_times, sff=unf_sff, csff=unf_csff)

        # Initialize spectral histogram
        plot = FormFactorPlot(data_path=path, unfold=False)

        # Plot spectral histogram
        plot.plot()

        # Initialize unfolded spectral histogram
        unf_plot = FormFactorPlot(data_path=unf_path, unfold=True)

        # Plot unfolded spectral histogram
        unf_plot.plot()

    def _worker_memory(self) -> float:
        """Calculate the memory required for each worker in GiB."""
        # Amount of memory required to store a random matrix
        matrix_memory = self.ensemble.matrix_memory

        # Amount of residual memory required to generate matrices
        resid_memory = self.ensemble.resid_memory

        # Amount of workspace memory required for each calculation
        work_memory = 36 * self.ensemble.dtype.itemsize * self.ensemble.dim

        # Return the total memory required for each worker in GiB
        return (matrix_memory + resid_memory + work_memory) / 2**30

    def _workspace_memory(self) -> float:
        """Calculate the actual workspace memory available in GiB."""
        # No shared memory, so return memory as is
        return self.memory


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
    # Avoid spawning issues on Windows
    set_start_method("fork", force=True)

    # Configure matplotlib
    configure_matplotlib()

    # Run main function
    main()
