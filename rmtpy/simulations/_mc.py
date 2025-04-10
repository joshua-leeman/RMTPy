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
        pass

    def _create_output_dir(self) -> None:
        pass

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
