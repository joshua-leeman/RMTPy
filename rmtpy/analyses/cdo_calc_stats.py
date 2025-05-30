# rmtpy.analyses.cdo_calc_stats.py


# =======================================
# 1. Imports
# =======================================
# Standard library imports
from __future__ import annotations
import os
import shutil
from argparse import ArgumentParser
from pathlib import Path

# Third-party imports
import numpy as np
from scipy.linalg import eigvalsh

# Local application imports
from rmtpy.simulations.cdo_evolve import construct_observable


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

    # Specify temporary output directory
    out_dir = os.path.join(data_dir, "stats")

    # Create temporary data directory for output
    os.makedirs(out_dir, exist_ok=True)

    # Store ID of this task
    process_id = os.environ.get("SLURM_PROCID", "0")

    # Update data directory specific to this process
    data_dir = os.path.join(data_dir, f"times_{process_id}")

    # Put all file names from directory into a list
    files = [
        str(file.resolve())
        for file in Path(data_dir).iterdir()
        if file.is_file() and file.suffix == ".npz"
    ]

    # Determine number of chunks based on number of files
    num_chunks = len(files)

    # Determine meta data of simulation from first file
    first_result = np.load(files[0])
    unfold = first_result["unfold"]
    N = first_result["N"]
    obs_q = first_result["obs_q"]
    r = first_result["realizs"]
    times = first_result["times"]
    dim = first_result["states"].shape[-1]
    dtype = first_result["states"].dtype
    real_dtype = first_result["states"].real.dtype
    del first_result

    # Calculate total number of realizations
    realizs = num_chunks * r

    # Initialize memory to store results
    states = np.empty((realizs, times.size, dim), dtype=dtype, order="F")

    # Loop through all files and load states
    for n, file in enumerate(files):
        # Load results
        results = np.load(file)

        # Store states in memory
        states[n * r : (n + 1) * r] = results["states"]

    # Construct observable
    obs = construct_observable(N, obs_q, dtype=dtype)

    # Calculate time evolution of observable expectation
    obs_expect = np.einsum("rtd,df,rtf->rt", states.conj(), obs, states, optimize=True)
    obs_expect = np.mean(obs_expect, axis=0).real

    # Compute squared observable in place
    np.matmul(obs, obs, out=obs)

    # Calculate variance of observable
    obs_var = np.einsum("rtd,df,rtf->rt", states.conj(), obs, states, optimize=True)
    obs_var = np.mean(obs_var, axis=0).real
    obs_var -= obs_expect**2

    # Delete observable to free memory
    del obs

    # Initialize memory for probabilities, purities, and entropy
    probs = np.empty((times.size, dim), dtype=real_dtype)
    c_purity = np.empty((times.size,), dtype=real_dtype)
    q_purity = np.empty((times.size,), dtype=real_dtype)
    entropy = np.empty((times.size,), dtype=real_dtype)

    # Transpose states for easier calculations
    states = states.transpose(1, 0, 2)

    # Initialize memory to store CDOs
    cdo = np.empty((dim, dim), dtype=dtype, order="F")

    # Loop through times to calculate probabilities, purities, and entropy
    for t in range(times.size):
        # Calculate chaotic density operator
        cdo = np.matmul(states[t].conj().T, states[t], out=cdo)
        cdo /= realizs

        # Calculate probabilities
        cdo_diag = np.diagonal(cdo).real
        probs[t, :] = cdo_diag

        # Calculate classical purity
        c_purity[t] = np.sum(cdo_diag**2)

        # Compute eigenvalues of CDO
        eigvals = eigvalsh(cdo, overwrite_a=True, check_finite=False)

        # Filter eigenvalues to avoid numerical issues
        eigvals = eigvals[eigvals > 1e-10]

        # Calculate quantum purity
        q_purity[t] = np.sum(eigvals**2)

        # Calculate von Neumann entropy
        entropy[t] = -np.sum(eigvals * np.log(eigvals)) if eigvals.size > 0 else 0.0

    # Create file name for output
    file_name = os.path.join(out_dir, f"{process_id}.npz")

    # Create temporary file name for output
    temp_file_name = os.path.join(out_dir, f"{process_id}.npz.tmp")

    # Save results to temporary file first
    with open(temp_file_name, "wb") as temp_file:
        # Store compressed data
        np.savez_compressed(
            temp_file,
            unfold=unfold,
            N=N,
            obs_q=obs_q,
            realizs=realizs,
            times=times,
            obs_expect=obs_expect,
            obs_var=obs_var,
            probs=probs,
            c_purity=c_purity,
            q_purity=q_purity,
            entropy=entropy,
        )

        # Flush file
        temp_file.flush()

        # Force write to disk
        os.fsync(temp_file.fileno())

    # Rename temporary file to final file name
    shutil.move(temp_file_name, file_name)


# If this script is run directly, execute main function
if __name__ == "__main__":
    # Run main function
    main()
