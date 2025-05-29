# rmtpy.analyses.cdo_evolve.py


# =======================================
# 1. Imports
# =======================================
# Standard library imports
from __future__ import annotations
import os
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

# Third-party imports
import numpy as np
from scipy.linalg import eigvalsh

# Local application imports
from rmtpy.simulations.cdo_evolve import construct_observable
from rmtpy.analyses._analysis import Analysis, _parse_analysis_args


# =======================================
# 2. CDO Evolution Analysis 2
# =======================================
@dataclass(repr=False, eq=False, frozen=True, kw_only=True, slots=True)
class CDOEvolve(Analysis):
    def run(self) -> None:
        # Put all file names from directory into a list
        files = [
            str(file.resolve())
            for file in Path(self.data_dir).iterdir()
            if file.is_file() and file.suffix == ".npz"
        ]

        # Load all data files into a list
        data = [np.load(file) for file in files]

        # From first file, extract meta data of simulation
        first_result = data[0]
        unfold = first_result["unfold"]
        N = first_result["N"]
        obs_q = first_result["obs_q"]
        realizs = first_result["realizs"]
        del first_result

        # Construct observable based on q-parameter
        observable = construct_observable(N, obs_q)

        # Compute eigenvalues of observable
        obs_eigvals = eigvalsh(observable, overwrite_a=True, check_finite=False)

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
        outdir = os.path.join(self.outdir, f"realizs_{realizs}")
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


# =======================================
# 3. Main Function
# =======================================
def main() -> None:
    # Create argument parser
    parser = ArgumentParser(description="CDO Evolution Analysis 2")

    # Parse command line arguments
    analysis_args = _parse_analysis_args(parser)

    # Create an instance of CDOEvolve class
    analysis = CDOEvolve(**analysis_args)

    # Run analysis
    analysis.run()


# If this script is run directly, execute main function
if __name__ == "__main__":
    # Run main function
    main()
