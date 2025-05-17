# rmtpy.analyses.spectral_statistics.py


# =======================================
# 1. Imports
# =======================================
# Standard library imports
from __future__ import annotations
import os
from argparse import ArgumentParser
from dataclasses import dataclass

# Third-party imports
import numpy as np

# Local application imports
from rmtpy.analyses._analysis import Analysis, _parse_analysis_args


# =======================================
# 2. Spectral Statistics Analysis
# =======================================
@dataclass(repr=False, eq=False, frozen=True, kw_only=True, slots=True)
class SpectralStatistics(Analysis):
    def run(self) -> None:
        """Run the spectral statistics analysis."""
        # Initialize final results with first .npz file
        results = np.load(self.files[0])

        # Store histogram bin edges and time arrays
        spec_bin_edges = results["spec_bin_edges"]
        unf_spec_bin_edges = results["unf_spec_bin_edges"]
        spac_bin_edges = results["spac_bin_edges"]
        unf_spac_bin_edges = results["unf_spac_bin_edges"]
        times = results["times"]
        unf_times = results["unf_times"]

        # Store first histogram and moments
        spec_hist = results["spec_hist"]
        unf_spec_hist = results["unf_spec_hist"]
        spac_hist = results["spac_hist"]
        unf_spac_hist = results["unf_spac_hist"]
        mu_1 = results["mu_1"]
        unf_mu_1 = results["unf_mu_1"]
        mu_2 = results["mu_2"]
        unf_mu_2 = results["unf_mu_2"]

        # Store number of realizations
        realizs = results["realizs"]

        # Loop through remaining .npz files
        for file in self.files[1:]:
            # Load results
            results = np.load(file)

            # Add histograms and moments
            spec_hist += results["spec_hist"]
            unf_spec_hist += results["unf_spec_hist"]
            spac_hist += results["spac_hist"]
            unf_spac_hist += results["unf_spac_hist"]
            mu_1 += results["mu_1"]
            unf_mu_1 += results["unf_mu_1"]
            mu_2 += results["mu_2"]
            unf_mu_2 += results["unf_mu_2"]

            # Delete results to free memory
            del results

        # Normalize histograms
        spec_hist /= np.sum(spec_hist * np.diff(spec_bin_edges))
        unf_spec_hist /= np.sum(unf_spec_hist * np.diff(unf_spec_bin_edges))
        spac_hist /= np.sum(spac_hist * np.diff(spac_bin_edges))
        unf_spac_hist /= np.sum(unf_spac_hist * np.diff(unf_spac_bin_edges))

        # Calculate form factors
        sff = mu_2 / realizs
        csff = sff - np.abs(mu_1) ** 2 / realizs**2
        unf_sff = unf_mu_2 / realizs
        unf_csff = unf_sff - np.abs(unf_mu_1) ** 2 / realizs**2

        # Append output directory with number of realizations
        outdir = os.path.join(self.outdir, f"realizs_{realizs}")

        # Create output directory if it doesn't exist
        os.makedirs(outdir, exist_ok=True)

        # Save results to file
        np.savez_compressed(
            os.path.join(outdir, "spectral_statistics.npz"),
            spec_bin_edges=spec_bin_edges,
            spec_hist=spec_hist,
            unf_spec_bin_edges=unf_spec_bin_edges,
            unf_spec_hist=unf_spec_hist,
            spac_bin_edges=spac_bin_edges,
            spac_hist=spac_hist,
            unf_spac_bin_edges=unf_spac_bin_edges,
            unf_spac_hist=unf_spac_hist,
            times=times,
            sff=sff,
            csff=csff,
            unf_times=unf_times,
            unf_sff=unf_sff,
            unf_csff=unf_csff,
        )


# =======================================
# 3. Main Function
# =======================================
def main() -> None:
    """Main function to run the spectral statistics simulation."""
    # Create argument parser
    parser = ArgumentParser(description="Spectral Statistics Monte Carlo Simulation")

    # Parse command line arguments
    mc_args = _parse_analysis_args(parser)

    # Create an instance of the SpectralStatistics class
    mc = SpectralStatistics(**mc_args)

    # Run simulation
    mc.run()


# If this script is run directly, execute main function
if __name__ == "__main__":
    # Run main function
    main()
