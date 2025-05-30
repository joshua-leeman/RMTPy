# rmtpy.analyses.spectral_statistics.py


# =======================================
# 1. Imports
# =======================================
# Standard library imports
from __future__ import annotations
import os
from argparse import ArgumentParser
from pathlib import Path

# Third-party imports
import numpy as np

# Local application imports
from rmtpy.utils import ensemble_from_path


# =======================================
# 2. Parse and Add Arguments
# =======================================
def parse_analysis_args(parser: ArgumentParser) -> dict:
    # Add data directory argument
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        required=True,
        help="data directory to analyze (required)",
    )

    # Add output directory argument
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        default=os.path.join(os.getcwd(), "output"),
        help="output directory (default is ./output)",
    )

    # Parse arguments into dictionary and return it
    return vars(parser.parse_args())


# =======================================
# 3. Main Analysis Function
# =======================================
def main() -> None:
    # Create argument parser
    parser = ArgumentParser(description="Spectral Statistics Analysis")

    # Parse command line arguments
    args = parse_analysis_args(parser)

    # Unpack analysis arguments
    data_dir = args["data_dir"]
    out_dir = args["out_dir"]

    # Put all file names from data directory into a list
    data_files = [
        str(file.resolve())
        for file in Path(data_dir).iterdir()
        if file.is_file() and file.suffix == ".npz"
    ]

    # Initialize ensemble from first file
    ensemble = ensemble_from_path(data_files[0])

    # Further specify output directory with simulation and ensemble
    out_dir = os.path.join(out_dir, "spectral_statistics", ensemble._to_path())

    # Initialize final results with first .npz file
    results = np.load(data_files[0])

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
    for file in data_files[1:]:
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
    outdir = os.path.join(out_dir, f"realizs_{realizs}")

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


# If this script is run directly, execute main function
if __name__ == "__main__":
    # Run main function
    main()
