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
# 2. Function
# =======================================
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
    num_times: int = 1000
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

    def _construct_observable(self) -> None:
        # Create Majorana pairs for observable
        majorana_pairs = create_majorana_pairs(self.ensemble.N)

        # Initialize observable matrix
        observable = np.zeros((self.ensemble.dim, self.ensemble.dim), order="F")

        # Retrieve indices for observable terms
        indices = tuple(itertools.combinations(range(self.ensemble.N), self.obs_q))

        # Loop over indices and fill observable matrix
        for idx in indices:
            # Divide indices into pairs
            pairs = tuple((idx[i], idx[i + 1]) for i in range(0, self.ensemble.q, 2))

            # Start q-body operator with first pair
            j0, k0 = pairs[0]
            q_body = majorana_pairs[j0][k0]

            # Multiply q-body operator with remaining pairs
            for j, k in pairs[1:]:
                q_body = q_body.dot(majorana_pairs[j][k])

            # Store q-body operator as COO matrix
            q_coo = q_body[: self.ensemble.dim, : self.ensemble.dim].tocoo()

            # Add q-body operator to observable matrix
            observable[q_coo.row, q_coo.col] += q_coo.data

        # Scale observable by necessary factors for hermicity and extensivity
        observable *= (
            1j ** (self.ensemble.q * (self.ensemble.q - 1) / 2)
            * np.sqrt(self.ensemble.N / comb(self.ensemble.N, self.ensemble.q))
            / np.log(2)
        )

        # Return observable matrix
        return observable

    def _construct_initial_state(self) -> None:
        # Construct observable matrix
        observable = self._construct_observable()

        # Calculate eigenvectors of observable
        eigvals, eigvecs = eigh(observable, overwrite_a=True, check_finite=False)

        # Calculate indices of sorted eigenvalues
        sorted_indices = np.argsort(eigvals)

        # Sort eigenvectors based on sorted eigenvalues
        eigvecs = eigvecs[:, sorted_indices]

        # Return eigenvector with largest eigenvalue as initial state
        return eigvecs[:, -1]

    def run(self) -> None:
        # Store first positive zero of 1st Bessel function
        j_1_1 = jn_zeros(1, 1)[0]

        # Create times array based on unfolding flag
        if self.unfold:
            # If unfolding is enabled, use unfolded times
            times = self.config._create_unf_times_array(base=self.ensemble.dim)
            times *= 2 * np.pi
        else:
            # If unfolding is disabled, use regular times
            times = self.config._create_times_array(base=self.ensemble.dim)
            times *= j_1_1 / self.ensemble.E0

        # Construct initial state
        initial_state = self._construct_initial_state()

        # Loop over diagonalization realizations and store evolved pure states
        for eigvals, eigvecs in self.ensemble.eig_stream(self.realizs):
            pass


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
