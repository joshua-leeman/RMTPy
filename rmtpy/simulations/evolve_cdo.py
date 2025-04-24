# rmtpy.simulations.evolve_cdo.py
"""
This module contains programs for performing Monte Carlo simulations to obtain the time evolutiovn of chaotic density operators (CDOs).
It is grouped into the following sections:
    1. Imports
    2. Plotting Functions
    3. Evolve CDO Class
    4. Main Function
"""


# =============================
# 1. Imports
# =============================
# Standard library imports
import os
import re
from argparse import ArgumentParser
from ast import literal_eval
from multiprocessing import Pool
from pathlib import Path
from textwrap import dedent
from time import time
from typing import Any, List

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import LogLocator, NullLocator
from psutil import virtual_memory

# Local application imports
from rmtpy.utils import get_ensemble, _create_plot, _initialize_plot
from rmtpy.simulations._mc import MonteCarlo
from rmtpy.configs import (
    probabilities_config,
    purity_config,
    entropy_config,
    expectation_config,
)


# =============================
# 2. Plotting Functions
# =============================
def plot_probabilities(data_path: str) -> None:
    pass


def plot_purity(data_path: str) -> None:
    pass


def plot_entropy(data_path: str) -> None:
    pass


def plot_expectation(data_path: str) -> None:
    pass


# =============================
# 3. Evolve CDO Class
# =============================
class EvolveCDO(MonteCarlo):
    """
    EvolveCDO class for performing Monte Carlo simulations to obtain the quantum-statistical time evolution of chaotic density operators (CDOs).
    Inherits from the MonteCarlo class.

    Methods
    -------
    run_simulation() -> None
        Runs a specific simulation and save the results.
    run() -> None
        Runs all specified simulations.
    """

    def __init__(
        self,
        ensemble: dict,
        realizations: int = 1,
        workers: int = 1,
        memory: int = virtual_memory().available // 2**30,
        runs: List[int] = [],
        unfold: List[int] = [],
    ) -> None:
        """
        Initialize the EvolveCDO simulation class.

        Parameters
        ----------
        ensemble : dict
            Ensemble parameters.
        realizations : int, optional
            Number of realizations (default is 1).
        workers : int, optional
            Number of workers (default is 1).
        memory : int, optional
            Memory allocated for simulation in bytes (default is total system memory).
        runs : list of int, optional
            List of simulations to run (default is empty list).
        unfold : list of int, optional
            List of simulations to unfold eigenvalues (default is empty list).

        Raises
        ------
        ValueError
            If unfold is not a subset of run or if 1 is included in unfold.
        """
        # Store runs and unfold arguments
        self._runs = runs
        self._unfold = unfold

        # Validate unfold is a subset of runs
        if not set(unfold).issubset(set(runs)):
            raise ValueError("Unfold must be a subset of run.")

        # Initialize Monte Carlo simulation
        super().__init__(ensemble, realizations, workers, memory)

        # If runs is empty, denote all flag and set run to all simulations
        if not runs:
            self._all = True
            self._runs = [1, 2, 3, 4]
        else:
            self._all = False

        # Store job arguments in dictionary
        self._job = {
            1: {
                "do": 1 in self._runs,
                "unfold": 1 in self._unfold,
                "func": self._spectral_hist,
                "plot": plot_probabilities,
                "file": probabilities_config.data_filename,
            },
            2: {
                "do": 2 in self._runs,
                "unfold": 2 in self._unfold,
                "func": self._nn_spacing_dist,
                "plot": plot_purity,
                "file": purity_config.data_filename,
            },
            3: {
                "do": 3 in self._runs,
                "unfold": 3 in self._unfold,
                "func": self._form_factors,
                "plot": plot_entropy,
                "file": entropy_config.data_filename,
            },
            4: {
                "do": 4 in self._runs,
                "unfold": 4 in self._unfold,
                "func": self._form_factors,
                "plot": plot_expectation,
                "file": expectation_config.data_filename,
            },
        }

    @staticmethod
    def _parse_args(parser: ArgumentParser) -> dict:
        """
        Parse command line arguments for the EvolveCDO simulation class.

        Parameters
        ----------
        parser : ArgumentParser
            Argument parser object.

        Returns
        -------
        dict
            Parsed arguments as a dictionary.
        """
        # Add arguments for which simulation(s) to run
        parser.add_argument(
            "-runs",
            "--runs",
            nargs="+",
            type=int,
            choices=[1, 2, 3, 4],
            default=[],
            help=dedent(
                """
                Specify which simulation(s) to run:
                    (1) Probabilities Evolution
                    (2) Purity Evolution
                    (3) Entropy Evolution
                    (4) Expectation Value Evolution
                """
            ),
        )

        # Add arguments for which simulation(s) to unfold eigenvalues
        parser.add_argument(
            "-unfold",
            "--unfold",
            nargs="+",
            type=int,
            choices=[1, 2, 3, 4],
            default=[],
            help=dedent(
                """
                Specify which simulation(s) to unfold eigenvalues (must be subset of --run and cannot be 1):
                    (1) Probabilities Evolution
                    (2) Purity Evolution
                    (3) Entropy Evolution
                    (4) Expectation Value Evolution
                """
            ),
        )

        # Send parser to Monte Carlo simulation class and return arguments
        return MonteCarlo._parse_args(parser)

    @staticmethod
    def _worker_func(args: dict) -> np.ndarray:
        pass

    def _realize_evolved_states(self) -> np.ndarray:
        """
        Divides times among workers and runs the simulation in parallel.

        Returns
        -------
        np.ndarray
            Sample of evolved states.
        """
        # Calculate typical size of each chunk


# =============================
# 4. Main Function
# =============================
def main() -> None:
    """
    Main function to run the CDO evolution Monte Carlo simulation from the command line.
    """
    # Create argument parser
    parser = ArgumentParser(description="Spectral Statistics Monte Carlo")

    # Retrieve Monte Carlo arguments
    mc_args = EvolveCDO._parse_args(parser)

    # Initialize spectral statistics simulation class
    mc = EvolveCDO(**mc_args)

    # Run spectral statistics simulation
    mc.run()


# Run the main function
if __name__ == "__main__":
    main()
