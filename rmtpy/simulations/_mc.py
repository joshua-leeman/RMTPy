# rmtpy.simulations._mc.py


# =======================================
# 1. Imports
# =======================================
# Standard library imports
from __future__ import annotations
import os
import re
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from ast import literal_eval
from dataclasses import dataclass, field
from typing import Any, Optional

# Local application imports
from rmtpy.utils import get_ensemble


# =======================================
# 2. Functions
# =======================================
def _parse_mc_args(parser: ArgumentParser) -> dict:
    # Add ensemble argument
    parser.add_argument(
        "-ens",
        "--ensemble",
        type=str,
        required=True,
        help="random matrix ensemble in JSON (required)",
    )

    # Add number of realizations argument
    parser.add_argument(
        "-r",
        "--realizs",
        type=int,
        default=1,
        help="number of realizations (default is 1)",
    )

    # Add output directory argument
    parser.add_argument(
        "-dir",
        "--outdir",
        type=str,
        default=os.path.join(os.getcwd(), "output"),
        help="output directory (default is current working directory)",
    )

    # Parse arguments into dictionary
    mc_args = vars(parser.parse_args())

    # Convert ensemble argument to dictionary
    mc_args["ensemble"] = literal_eval(mc_args["ensemble"])

    # Return dictionary of Monte Carlo arguments
    return mc_args


# =======================================
# 3. Monte Carlo Simulation Class
# =======================================
@dataclass(repr=False, eq=False, frozen=True, kw_only=True, slots=True)
class MonteCarlo(ABC):
    # Random matrix ensemble
    ensemble: dict

    # Number of realizations
    realizs: int

    # Output directory
    outdir: str

    # Configuration object
    config: Optional[Any] = None

    @abstractmethod
    def run(self) -> None:
        """Run the Monte Carlo simulation."""
        pass

    def __post_init__(self):
        """Post-initialization method to set up the Monte Carlo simulation."""
        # Initialize ensemble
        ensemble = get_ensemble(self.ensemble)

        # Store ensemble
        object.__setattr__(self, "ensemble", ensemble)

        # Further specify output directory
        self._create_outdir(self.outdir)

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attribute access to the configuration."""
        # Store class name
        class_name = self.__class__.__name__

        # Check if config is set
        if self.config is None:
            raise AttributeError(f"'{class_name}' object has no attribute '{name}'")

        # Check if attribute exists in config
        if hasattr(self.config, name):
            return getattr(self.config, name)

        # Raise error if attribute does not exist
        raise AttributeError(f"'{class_name}' object has no attribute '{name}'")

    def _create_outdir(self, outdir: str) -> str:
        """Create the finalized output directory for the simulation."""
        # Replace underscores in class name with spaces
        mc_path = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", self.__class__.__name__)
        mc_path = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", mc_path)
        mc_path = mc_path.lower()

        # Further specify output directory with simulation and ensemble
        outdir = os.path.join(outdir, mc_path, self.ensemble._to_path())

        # Append number of realizations to output directory
        outdir = os.path.join(outdir, f"realizs_{self.realizs}")

        # Create output directory if it does not exist
        os.makedirs(outdir, exist_ok=True)

        # Store output directory
        object.__setattr__(self, "outdir", outdir)
