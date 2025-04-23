# rmtpy.simulations._mc.py
"""
This module contains the Monte Carlo simulation base class for the RMTpy package.
It is grouped into the following sections:
    1. Imports
    2. Monte Carlo Class
"""


# =============================
# 1. Imports
# =============================
# Standard library imports
import os
import re
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from ast import literal_eval
from textwrap import dedent
from time import strftime

# Third-party imports
from psutil import cpu_count, virtual_memory

# Local application imports
from rmtpy.utils import get_ensemble


# =============================
# 2. Monte Carlo Class
# =============================
class MonteCarlo(ABC):
    """
    Monte Carlo simulation base class.

    Attributes
    ----------
    ensemble : object
        Random matrix ensemble object.
    realizs : int
        Number of realizations.
    max_workers : int
        Maximum number of workers available.
    max_memory : int
        Maximum memory available in bytes.
    workers : int
        Number of workers for simulation.
    memory : int
        Memory allocated for simulation in bytes.
    calc_memory : int
        Memory required for calculations in bytes.

    Methods
    -------
    run()
        Run the Monte Carlo simulation.
    """

    def __init__(
        self,
        ensemble: dict,
        realizs: int = 1,
        workers: int = 1,
        memory: float = (0.9 * virtual_memory().total) // 2**30,
    ) -> None:
        """
        Initialize the Monte Carlo simulation.

        Parameters
        ----------
        ensemble : dict
            Random matrix ensemble parameters.
        realizs : int, optional
            Number of realizations (default is 1).
        workers : int, optional
            Number of workers (default is 1).
        memory : float, optional
            Memory allocated for simulation in GiB (default is 90% of total memory).
        """
        # Clean ensemble name in ens_args
        ensemble["name"] = re.sub(r"\W+", "", ensemble["name"]).strip().lower()

        # Copy ensemble input and pop name
        ens_inputs = ensemble.copy()
        ens_inputs.pop("name")

        # Import and initialize ensemble
        self._ensemble = get_ensemble(ensemble["name"], **ens_inputs)

        # Reorder ensemble input and and store
        self._ens_args = {
            key: ensemble[key] for key in self.ensemble._arg_order if key in ensemble
        }

        # Store number of realizations
        self._realizs = realizs

        # Store system specifications
        self._max_workers = cpu_count(logical=False)
        self._max_memory = virtual_memory().total  # in bytes

        # Store job specifications
        self._workers = workers
        self._memory = memory  # in GiB

        # Check if Monte Carlo simulation is valid
        self._check_mc()

        # Store date and time of simulation
        self._time_date = strftime("%Y-%m-%d %H:%M:%S")
        self._time_path = self._time_date.replace(" ", "_").replace(":", "-")

    def __repr__(self) -> str:
        """
        Default representation of the Monte Carlo simulation.
        """
        return f"{self.__class__.__name__} ({self.ensemble}, realizs={self.realizs})"

    def __str__(self) -> str:
        """
        Path representation of the Monte Carlo simulation.
        """
        # Replace underscores with spaces
        string = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", self.__class__.__name__)
        string = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", string)

        # Convert to lowercase and return
        return string.lower()

    @staticmethod
    def _parse_args(parser: ArgumentParser) -> dict:
        """
        Get Monte Carlo arguments from the parser.

        Parameters
        ----------
        parser : ArgumentParser
            Parser for command-line options.

        Returns
        -------
        dict
            Monte Carlo arguments.
        """
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
            "--realizations",
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

        # Store total memory in GiB
        total_memory = virtual_memory().total // 2**30

        # Add memory argument
        parser.add_argument(
            "-m",
            "--memory",
            type=int,
            default=total_memory,
            help=f"memory allocated for simulation in GiB (default is {total_memory})",
        )

        # Parse arguments into dictionary
        mc_args = vars(parser.parse_args())

        # Convert ensemble argument to dictionary
        mc_args["ensemble"] = literal_eval(mc_args["ensemble"])

        # Return dictionary of Monte Carlo arguments
        return mc_args

    def _check_mc(self) -> None:
        """
        Check if Monte Carlo simulation is valid.

        Raises
        ------
        ValueError
            If the Monte Carlo simulation is not valid and explains why.
        """
        # Check if number of realizations is valid
        if (
            not isinstance(self._realizs, (int, float))
            or self._realizs < 1
            or self._realizs != int(self._realizs)
        ):
            raise ValueError(f"Number of realizations must be a positive integer.")

        # Check if number of workers is valid
        # If so, set to int
        if (
            not isinstance(self._workers, (int, float))
            or self._workers < 1
            or self._workers != int(self._workers)
            or self._workers > self._max_workers  # type: ignore[operator]
        ):
            raise ValueError(
                f"Number of workers must be a positive integer less than or equal to {self._max_workers}."
            )
        else:
            self._workers = int(self._workers)

        # Check if memory is valid
        # If so, set to bytes
        if (
            not isinstance(self._memory, (int, float))
            or self._memory < 1
            or self._memory > self._max_memory // 2**30
        ):
            raise ValueError(
                f"Memory must be a positive integer less than or equal to {self.max_memory / 2**30:.1f} [in GiB]."
            )
        else:
            self._memory = self.memory * 2**30

        # Check if provided memory is sufficient
        # If so, determine number of usable workers
        if self.memory < self.calc_memory:
            raise ValueError(
                dedent(
                    f"""
                    Provided memory is insufficient for calculations:
                    Memory Required > {self.calc_memory // 2**30} GB
                    Memory Provided: {self.memory // 2**30} GB
                    """
                )
            )
        else:
            self._workers = min(self.workers, self.memory // self.calc_memory)

    def _create_output_dir(self, res_type: str = "") -> str:
        """
        Returns the project's results directory for the Monte Carlo simulation.

        Parameters
        ----------
        res_type : str, optional
            Type of results directory. (default is "")

        Returns
        -------
        str
            Results directory path.
        """
        # Construct results directory path
        output_dir = f"res/{str(self)}/{self._ens_args['name']}/"
        output_dir += "/".join(
            f"{key}={val}" for key, val in self._ens_args.items() if key != "name"
        )
        output_dir += f"/realizs={self.realizs}/{self._time_path}/{res_type}"

        # Create directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)

        # Return results directory path
        return output_dir

    @abstractmethod
    def run(self) -> None:
        """
        Run the specified Monte Carlo simulation(s).
        """
        pass

    @property
    def ensemble(self):
        """
        Ensemble object.
        """
        return self._ensemble

    @property
    def realizs(self):
        """
        Number of realizations.
        """
        return self._realizs

    @property
    def max_workers(self):
        """
        Maximum number of workers.
        """
        return self._max_workers

    @property
    def max_memory(self):
        """
        Maximum memory available. [bytes]
        """
        return self._max_memory

    @property
    def workers(self):
        """
        Number of workers for simulation.
        """
        return self._workers

    @property
    def memory(self):
        """
        Memory allocated for simulation. [bytes]
        """
        return self._memory

    @property
    def calc_memory(self):
        """
        Memory required for calculations. [bytes]
        """
        return 4 * self._ensemble.matrix_memory
