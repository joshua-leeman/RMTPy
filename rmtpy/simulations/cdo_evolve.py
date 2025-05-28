# rmtpy.simulations.cdo_evolve.py


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
# 2. Function
# =======================================
def _parse_cdo_evolve_args(parser: ArgumentParser) -> dict:
    # Add unfolding flag argument
    parser.add_argument(
        "-unf",
        "--unfolding",
        action="store_true",
        help="flag to indicate unfolding of energies (default is False)",
    )

    # Add observable q-parameter argument
    parser.add_argument(
        "-obs_q",
        "--obs_q_param",
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
    # Simulation configuration
    config: Config = field(default_factory=Config)


# =======================================
# 5. Main Function
# =======================================
def main() -> None:
    # Create argument parser
    parser = ArgumentParser(
        description="Run Chaotic Density Operator Evolution Simulation."
    )

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
