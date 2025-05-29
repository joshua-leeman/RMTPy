# rmtpy.simulations.cdo_evolve.py


# =======================================
# 1. Imports
# =======================================
# Standard library imports
from __future__ import annotations
import itertools
import os
import shutil
from argparse import ArgumentParser
from dataclasses import dataclass, field
from math import comb

# Third-party imports
import numpy as np
from scipy.linalg import eigh
from scipy.special import jn_zeros

# Local application imports
from rmtpy.simulations._mc import MonteCarlo, _parse_mc_args
from rmtpy.special import create_majorana_pairs


# =======================================
# 2. Functions
# =======================================
def construct_observable(
    N: int, obs_q: int, dtype: np.dtype = np.complex128
) -> np.ndarray:
    # Calculate dimension of observable matrix
    dim = 2 ** (N // 2 - 1)

    # Create Majorana pairs for observable
    majorana_pairs = create_majorana_pairs(N)

    # Initialize observable matrix
    observable = np.zeros((dim, dim), dtype=dtype, order="F")

    # Retrieve indices for observable terms
    indices = tuple(itertools.combinations(range(N), obs_q))

    # Loop over indices and fill observable matrix
    for idx in indices:
        # Divide indices into pairs
        pairs = tuple((idx[i], idx[i + 1]) for i in range(0, obs_q, 2))

        # Start q-body operator with first pair
        j0, k0 = pairs[0]
        q_body = majorana_pairs[j0][k0]

        # Multiply q-body operator with remaining pairs
        for j, k in pairs[1:]:
            q_body = q_body.dot(majorana_pairs[j][k])

        # Store q-body operator as COO matrix
        q_coo = q_body[:dim, :dim].tocoo()

        # Add q-body operator to observable matrix
        observable[q_coo.row, q_coo.col] += q_coo.data

    # Scale observable by necessary factors for hermicity and extensivity
    observable *= (
        1j ** (obs_q * (obs_q - 1) / 2) * np.sqrt(N / comb(N, obs_q)) / np.log(2)
    )

    # Return observable matrix
    return observable


def construct_initial_state(
    N: int, obs_q: int, dtype: np.dtype = np.complex128
) -> None:
    # Construct observable matrix
    observable = construct_observable(N, obs_q, dtype=dtype)

    # Calculate eigenvectors of observable
    _, eigvecs = eigh(observable, overwrite_a=True, check_finite=False)

    # Return eigenvector with largest eigenvalue as initial state
    return eigvecs[:, -1]


def _parse_cdo_evolve_args(parser: ArgumentParser) -> dict:
    # Add unfolding flag argument
    parser.add_argument(
        "-unf",
        "--unfold",
        action="store_true",
        help="flag to indicate unfolding of energies (default is False)",
    )

    # Add observable q-parameter argument
    parser.add_argument(
        "-q",
        "--obs_q",
        type=int,
        default=2,
        help="observable q-parameter (default is 2)",
    )

    # Send parser to _parse_mc_args to add common arguments and return arguments
    return _parse_mc_args(parser)


# =======================================
# 3. Configuration Class
# =======================================
@dataclass(repr=False, eq=False, frozen=True, kw_only=True, slots=True)
class Config:
    # Simulation parameters
    num_times: int = 500
    logtime_i: float = -0.5  # base = dim
    logtime_f: float = 1.5  # base = dim
    unf_logtime_i: float = -1.5  # base = dim
    unf_logtime_f: float = 0.5  # base = dim

    def _create_times_array(self, base: float) -> np.ndarray:
        """Create times array."""
        # Create logarithmically spaced array of times and return it
        return np.logspace(self.logtime_i, self.logtime_f, self.num_times, base=base)

    def _create_unf_times_array(self, base: float) -> np.ndarray:
        """Create unfolded times array."""
        # Create logarithmically spaced array of unfolded times and return it
        return np.logspace(
            self.unf_logtime_i, self.unf_logtime_f, self.num_times, base=base
        )


# =======================================
# 4. Simulation Class
# =======================================
@dataclass(repr=False, eq=False, frozen=True, kw_only=True, slots=True)
class CDOEvolve(MonteCarlo):
    # Unfolding flag
    unfold: bool = False

    # Observable q-parameter
    obs_q: int = 2

    # Simulation configuration
    config: Config = field(default_factory=Config)

    def run(self) -> None:
        # Create times array based on unfolding flag
        if self.unfold:
            # If unfolding is enabled, use unfolded times
            times = self.config._create_unf_times_array(base=self.ensemble.dim)
            times *= 2 * np.pi
        else:
            # Store first positive zero of 1st Bessel function
            j_1_1 = jn_zeros(1, 1)[0]

            # If unfolding is disabled, use regular times
            times = self.config._create_times_array(base=self.ensemble.dim)
            times *= j_1_1 / self.ensemble.E0

        # Construct initial state
        initial_state = construct_initial_state(
            self.ensemble.N, self.obs_q, dtype=self.ensemble.dtype
        )

        # Initialize array to store evolved states
        evolved_states = np.empty(
            (self.realizs, times.size, self.ensemble.dim),
            dtype=self.ensemble.dtype,
            order="F",
        )

        # Loop over diagonalization realizations and store evolved pure states
        for r, eigsys in enumerate(self.ensemble.eig_stream(self.realizs)):
            # Unpack eigenvalues and eigenvectors
            eigvals, eigvecs = eigsys

            # If unfolding is enabled, unfold eigenvalues
            if self.unfold:
                eigvals = self.ensemble.unfold(eigvals)

            # Rotate initial state into eigenbasis
            rotated_state = np.matmul(eigvecs.T.conj(), initial_state)

            # Outer-multiply eigenvalues and times, exponentiate, then broadcast multiply
            np.outer(times, eigvals, out=evolved_states[r])
            evolved_states[r] *= -1j
            np.exp(evolved_states[r], out=evolved_states[r])
            np.multiply(evolved_states[r], rotated_state, out=evolved_states[r])

            # Rotate evolved states back to original basis
            np.matmul(evolved_states[r], eigvecs.T, out=evolved_states[r])

        # Create indices to chunk evolved states over times
        ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
        chunk_sizes = [(times.size + i) // ntasks for i in range(ntasks)]
        indices = np.cumsum([0] + chunk_sizes)

        # Store views of times for each task
        times_chunks = [
            times[indices[i] : indices[i + 1]] for i in range(len(indices) - 1)
        ]

        # Store views of evolved states for each task
        state_chunks = [
            evolved_states[:, indices[i] : indices[i + 1], :]
            for i in range(len(indices) - 1)
        ]

        # Store ID of this task
        process_id = int(os.environ.get("SLURM_PROCID", "0"))

        # Loop over tasks and store results
        for n in range(len(times_chunks)):
            # Create output directory for chunk of times
            outdir = os.path.join(self.outdir, f"times_{n}")
            os.makedirs(outdir, exist_ok=True)

            # Create file name for output
            file_name = os.path.join(outdir, f"{process_id}.npz")

            # Create temporary file name for output
            temp_file_name = os.path.join(outdir, f"{process_id}.npz.tmp")

            # Save results to temporary file first
            with open(temp_file_name, "wb") as temp_file:
                # Store compressed data
                np.savez_compressed(
                    temp_file,
                    unfold=self.unfold,
                    N=self.ensemble.N,
                    obs_q=self.obs_q,
                    realizs=self.realizs,
                    times=times_chunks[n],
                    states=state_chunks[n],
                )

                # Flush file
                temp_file.flush()

                # Forece write to disk
                os.fsync(temp_file.fileno())

            # Rename temporary file to final file name
            shutil.move(temp_file_name, file_name)


# =======================================
# 5. Main Function
# =======================================
def main() -> None:
    # Create argument parser
    parser = ArgumentParser(description="CDO Evolution Monte Carlo Simulation.")

    # Parse command line arguments
    cdo_evolve_args = _parse_cdo_evolve_args(parser)

    # Create CDOEvolve simulation instance
    mc = CDOEvolve(**cdo_evolve_args)

    # Run simulation
    mc.run()


# If this script is run directly, execute main function
if __name__ == "__main__":
    # Run main function
    main()
