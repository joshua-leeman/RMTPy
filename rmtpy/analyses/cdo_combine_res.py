# rmtpy.analyses.cdo_combine_res.py


# rmtpy.analyses.cdo_calc_stats.py


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
from scipy.linalg import eigvalsh

# Local application imports
from rmtpy.simulations.cdo_evolve import construct_observable
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
    parser = ArgumentParser(description="Calculate CDO Statistics")

    # Parse command line arguments
    args = parse_analysis_args(parser)

    # Unpack analysis arguments
    data_dir = args["data_dir"]
    out_dir = args["out_dir"]

    # Put all file names from directory into a list
    data_files = [
        str(file.resolve())
        for file in Path(data_dir).iterdir()
        if file.is_file() and file.suffix == ".npz"
    ]

    # Initialize ensemble from first file
    ensemble = ensemble_from_path(data_files[0])

    # Further specify output directory with simulation and ensemble
    out_dir = os.path.join(out_dir, "cdo_evolve", ensemble._to_path())

    # Load all data files into a list
    data = [np.load(file) for file in data_files]

    # From first file, extract meta data of simulation
    first_result = data[0]
    unfold = first_result["unfold"]
    N = first_result["N"]
    obs_q = first_result["obs_q"]
    realizs = first_result["realizs"]
    del first_result

    # Construct observable based on q-parameter
    obs = construct_observable(N, obs_q)

    # Compute eigenvalues of observable
    obs_eigvals = eigvalsh(obs, overwrite_a=True, check_finite=False)

    # Delete observable to free memory
    del obs

    # Compute indices of ordered time data
    times = np.concatenate([d["times"] for d in data])
    ordered_indices = np.argsort(times)

    # Sort expectation values based on ordered indices
    obs_expect = np.concatenate([d["obs_expect"] for d in data])
    obs_expect = obs_expect[ordered_indices]

    # Sort variances based on ordered indices
    obs_var = np.concatenate([d["obs_var"] for d in data])
    obs_var = obs_var[ordered_indices]

    # Sort probabilities based on ordered indices
    probs = np.concatenate([d["probs"] for d in data])
    probs = probs[ordered_indices]

    # Sort classical purity values based on ordered indices
    c_purity = np.concatenate([d["c_purity"] for d in data])
    c_purity = c_purity[ordered_indices]

    # Sort quantum purity values based on ordered indices
    q_purity = np.concatenate([d["q_purity"] for d in data])
    q_purity = q_purity[ordered_indices]

    # Sort entropy values based on ordered indices
    entropy = np.concatenate([d["entropy"] for d in data])
    entropy = entropy[ordered_indices]

    # Append output directory with number of realizations
    outdir = os.path.join(out_dir, f"realizs_{realizs}")
    os.makedirs(outdir, exist_ok=True)

    # Store compressed data
    np.savez_compressed(
        os.path.join(outdir, "cdo_evolve.npz"),
        unfold=unfold,
        N=N,
        obs_q=obs_q,
        realizs=realizs,
        times=times,
        obs_eigvals=obs_eigvals,
        obs_expect=obs_expect,
        obs_var=obs_var,
        probs=probs,
        c_purity=c_purity,
        q_purity=q_purity,
        entropy=entropy,
    )


# If this script is run directly, execute main function
if __name__ == "__main__":
    # Run main function
    main()
