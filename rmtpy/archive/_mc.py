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
from textwrap import dedent
from time import strftime
from typing import Any, Optional, Union

# Third-party imports
from psutil import cpu_count, virtual_memory

# Local application imports
from rmtpy.utils import get_ensemble


# =======================================
# 2. Functions
# =======================================
def _parse_mc_args(parser: ArgumentParser) -> dict:
    """Parse default command line arguments."""
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
        "-realizs",
        "--realizs",
        type=int,
        default=1,
        help="number of realizations (default is 1)",
    )

    # Add number of workers argument
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=1,
        help=f"number of workers (default is 1, maximum is {cpu_count(logical=False)})",
    )

    # Store available and total memory in GiB
    avail_memory = virtual_memory().available / 2**30

    # Add memory argument
    parser.add_argument(
        "-m",
        "--memory",
        type=int,
        default=avail_memory,
        help=f"memory allocated for simulation in GiB (default is {avail_memory})",
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
@dataclass(repr=False, eq=False, kw_only=True, slots=True)
class MonteCarlo(ABC):
    # Random matrix ensemble
    ensemble: Union[Any, dict, str]

    # Number of realizations
    realizs: int = field(default=1)

    # Amount of memory to allocate for simulation in GiB
    memory: float = field(default=virtual_memory().available / 2**30)

    # Maximum amount of system memory
    max_memory: float = field(default=virtual_memory().total / 2**30)

    # Number of workers
    workers: int = field(default=1)

    # Output directory
    output_dir: Optional[str] = field(default=None)

    # Date and time of simulation
    time_str: Optional[str] = field(init=False, default=strftime("%Y-%m-%d %H:%M:%S"))

    # Maxumum number of workers available
    max_workers: int = field(init=False, default=cpu_count(logical=False))

    # Memory required for each worker in bytes
    worker_memory: float = field(init=False, default=None)

    @abstractmethod
    def run(self) -> None:
        """Run the Monte Carlo simulation."""
        raise NotImplementedError("run method must be implemented in subclasses.")

    @abstractmethod
    def _worker_memory(self) -> float:
        """Calculate the memory required for each calculation in GiB."""
        raise NotImplementedError("_worker_memory must be implemented in subclass.")

    @abstractmethod
    def _workspace_memory(self) -> float:
        """Calculate the actual workspace memory is available in GiB."""
        raise NotImplementedError("_workspace_memory must be implemented in subclass.")

    def __post_init__(self) -> None:
        # If ensemble is a dictionary or string, convert it to an ensemble object
        if isinstance(self.ensemble, (dict, str)):
            self.ensemble = get_ensemble(self.ensemble)

        # Store amount of memory required for each calculation
        self.worker_memory = self._worker_memory()

        # Store amount of actual workspace memory available
        self.memory = self._workspace_memory()

        # Check if Monte Carlo simulation is valid
        self._check_mc()

        # Reduce number of workers if necessary
        self.workers = min(self.workers, self.memory // self.worker_memory)

        # Create output directory
        self.output_dir = self._create_output_dir()

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attribute access to the ensemble."""
        # Return attribute from ensemble if it exists
        return getattr(self.ensemble, name)

    def _check_mc(self) -> None:
        """Check if Monte Carlo simulation is valid."""
        # Check if number of realizations is valid
        if not isinstance(self.realizs, int) or self.realizs < 1:
            raise ValueError("Number of realizations must be a positive integer.")

        # Check if number of workers is valid
        if not isinstance(self.workers, int) or self.max_workers < self.workers < 1:
            raise ValueError(
                f"Number of workers must be a positive integer between 1 and {self.max_workers}."
            )

        # Check if memory is valid
        if (
            not isinstance(self.memory, (int, float))
            or self.max_memory < self.memory < 0
        ):
            raise ValueError(
                f"Memory must be a positive integer or float between 0 and {self.max_memory} GiB."
            )

        # Check if provided memory is sufficient for calculations
        if self.worker_memory > self.memory:
            raise MemoryError(
                dedent(
                    f"""
                    Not enough memory available for calculations:
                    Required: {self.worker_memory:.2f} GiB
                    Requested: {self.memory:.2f} GiB
                    """
                )
            )

    def _create_output_dir(self) -> str:
        """Create the output directory for the simulation."""
        # Replace underscores in class name with spaces
        mc_path = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", self.__class__.__name__)
        mc_path = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", mc_path)
        mc_path = mc_path.lower()

        # Begin output directory with outputs, simulation, and ensemble path
        output_dir = f"outputs/{mc_path}/{self.ensemble._to_path()}"

        # Translate time string to a valid directory name
        time_path = self.time_str.replace(" ", "_").replace(":", "-")

        # Append simulation attributes, titme, and results type to output directory
        output_dir += f"/realizs={self.realizs}/{time_path}/data"

        # Create output directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)

        # Return output directory
        return output_dir
