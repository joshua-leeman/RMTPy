# rmtpy.analyses._analysis.py


# =======================================
# 1. Imports
# =======================================
# Standard library imports
from __future__ import annotations
import os
import re
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from dataclasses import dataclass

# Local application imports
from rmtpy.utils import ensemble_from_path


# =======================================
# 2. Functions
# =======================================
def _parse_analysis_args(parser: ArgumentParser) -> dict:
    # Add list of files argument
    parser.add_argument(
        "-f",
        "--files",
        type=str,
        nargs="+",
        required=True,
        help="list of files to analyze (required)",
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
    args = vars(parser.parse_args())

    # Return parsed arguments
    return args


# =======================================
# 3. Analysis Class
# =======================================
@dataclass(repr=False, eq=False, frozen=True, kw_only=True, slots=True)
class Analysis(ABC):
    # List of files to analyze
    files: list[str]

    # Output directory
    outdir: str = os.path.join(os.getcwd(), "output")

    @abstractmethod
    def run(self) -> None:
        """Run the analysis."""
        pass

    def __post_init__(self) -> None:
        """Post-initialization to create finalized output directory"""
        # Replace underscores in class name with spaces
        mc_path = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", self.__class__.__name__)
        mc_path = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", mc_path)
        mc_path = mc_path.lower()

        # Initialize ensemble from first file
        ensemble = ensemble_from_path(self.files[0])

        # Further specify output directory with simulation and ensemble
        outdir = os.path.join(self.outdir, mc_path, ensemble._to_path())

        # Store output directory
        object.__setattr__(self, "outdir", outdir)
