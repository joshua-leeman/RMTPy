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
from importlib import import_module
from textwrap import dedent
from time import strftime
from typing import Tuple

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from psutil import cpu_count, virtual_memory

# Load environment variables
load_dotenv()


# =============================
# 2. Monte Carlo Class
# =============================
class MonteCarlo(ABC):
    def __init__(self, ens_args: dict, sim_args: dict, spec_args: dict = {}) -> None:
        # Clean ensemble name in ens_args
        ens_args["name"] = re.sub(r"\W+", "", ens_args["name"]).strip().lower()

        # Copy ensemble input and pop name
        ens_args = ens_args.copy()
        ens_args.pop("name")

        # Initialize ensemble
        module = import_module(f"rmtpy.ensembles.{ens_args['name']}")
        ENSEMBLE = getattr(module, module.class_name)
        self._ensemble = ENSEMBLE(**ens_args)

        # Reorder ensemble input and and store
        self._ens_args = {
            key: ens_args[key] for key in self.ensemble._arg_order if key in ens_args
        }

        # Store simulation input and update with name
        self._sim_input = sim_args.copy()
        self._sim_input["name"] = str(self)

        # Store number of realizations
        self._realizs = int(sim_args.get("realizations", 1))

        # Store system specifications
        self._max_workers = cpu_count(logical=False)
        self._max_memory = virtual_memory().total  # in bytes

        # Store job specifications
        self._workers = spec_args.get("workers", 1)
        self._memory = spec_args.get("memory", self.max_memory // 2**30)  # in GiB

        # Check if Monte Carlo simulation is valid
        self._check_mc()

        # Store project path
        self._project_path = os.getenv("PROJECT_PATH")

        # Store date and time of simulation
        self._time_date = strftime("%Y-%m-%d %H:%M:%S")
        self._time_path = self._time_date.replace(" ", "_").replace(":", "-")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} ({self.ensemble}, realizs={self.realizs})"

    def __str__(self) -> str:
        # Replace underscores with spaces
        string = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", self.__class__.__name__)
        string = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", string)

        # Convert to lowercase and return
        return string.lower()

    @staticmethod
    def _parse_args(parser: ArgumentParser) -> dict:
        # Add ensemble argument
        parser.add_argument(
            "-ens",
            "--ensemble",
            type=str,
            required=True,
            help="random matrix ensemble in JSON (required)",
        )

        # Add simulation argument(s)
        parser.add_argument(
            "-args",
            "--arguments",
            type=str,
            default="{'realizs': 1}",
            help="simulation arguments in JSON (default is {'realizs': 1})",
        )

        # Store total memory in GiB
        total_memory = virtual_memory().total // 2**30

        # Add job specification argument(s)
        parser.add_argument(
            "-spec",
            "--specification",
            type=str,
            default=f"{{'workers': 1, 'memory': {total_memory}}}",
            help=f"job specification in JSON (default is {{'workers': 1, 'memory': {total_memory} [in GiB]}})",
        )

        # Parse arguments into dictionary
        parsed_args = vars(parser.parse_args())

        # Initialize output dictionary
        mc_args = {}

        # Convert ensemble argument to dictionary
        mc_args["ens_args"] = literal_eval(parsed_args["ensemble"])

        # Convert simulation argument to dictionary
        mc_args["sim_args"] = literal_eval(parsed_args["arguments"])

        # Convert job specification argument to dictionary
        mc_args["spec_args"] = literal_eval(parsed_args["specification"])

        # Return dictionary of Monte Carlo arguments
        return mc_args

    def _check_mc(self) -> None:
        # Retrieve list of valid ensembles
        ensemble_list = [
            file.rstrip(".py")
            for file in os.listdir(f"{self._project_path}/rmtpy/ensembles")
            if file.endswith(".py") and not file.startswith("_")
        ]

        # Check if ensemble is valid
        if self._ens_input["name"] not in ensemble_list:
            raise ValueError(
                dedent(
                    f"""
                    Ensemble '{self._ens_input["name"]}' is not valid.
                    Valid ensembles are: {ensemble_list}
                    """
                )
            )

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
            or self._workers > self._max_workers
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
        # Construct results directory path
        output_dir = f"{self.project_path}/res/{str(self)}/{self._ens_args['name']}/"
        output_dir += "/".join(
            f"{key}_{val}" for key, val in self._ens_args.items() if key != "name"
        )
        output_dir += f"/realizs={self.realizs}/{self._time_path}/{res_type}"

        # Create directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)

        # Return results directory path
        return output_dir

    @abstractmethod
    def run(self) -> None:
        pass

    @property
    def ensemble(self):
        return self._ensemble

    @property
    def realizs(self):
        return self._realizs

    @property
    def max_workers(self):
        return self._max_workers

    @property
    def max_memory(self):
        return self._max_memory

    @property
    def workers(self):
        return self._workers

    @property
    def memory(self):
        return self._memory

    @property
    def calc_memory(self):
        return 4 * self._ensemble.matrix_memory
