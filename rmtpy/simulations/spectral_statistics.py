# rmtpy.simulations.spectral_statistics.py
"""
This module contains programs for performing Monte Carlo simulations to obtain the spectral statistics of random matrix ensembles.
It is grouped into the following sections:
    1. Imports
    2. Functions
    3. Spectral Statistics Class
    4. Main Function
"""


# =============================
# 1. Imports
# =============================
# Standard library imports
import os
from argparse import ArgumentParser
from importlib import import_module
from multiprocessing import Pool
from time import time
from typing import Dict, List

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt

# Local application imports
from rmtpy.simulations._mc import MonteCarlo
from rmtpy.configs.spectral_statistics_config import (
    spectral_config,
    spacings_config,
    sff_config,
)


# =============================
# 2. Functions
# =============================
def _create_histogram(data: np.ndarray, dataclass: object, path: str = "") -> None:
    # Create default path if not provided
    if path == "":
        path = f"res/data/{dataclass.data_filename}"

    # Calculate bin edges
    min_edge, max_edge = np.min(data), np.max(data)

    # Arrange bin edges
    bins = np.arange(min_edge, max_edge + dataclass.bin_width, dataclass.bin_width)

    # Calculate normalized histogram of data
    hist_counts, hist_edges = np.histogram(data, bins=bins, density=True)

    # Save histogram data
    if path.endswith(".npz"):
        np.savez_compressed(
            path,
            hist_counts=hist_counts,
            hist_edges=hist_edges,
        )
    elif path.endswith(".csv"):
        np.savetxt(
            path,
            np.column_stack((hist_edges[:-1], hist_counts)),
            delimiter=",",
            header="Bin Edges, Counts",
            comments="",
        )
    else:
        raise ValueError(f"Unsupported file type. Expected .npz or .csv, got {path}")


def calc_spectral_hist(eigenvalues: np.ndarray, path: str = "") -> None:
    # Create and save histogram of eigenvalues
    _create_histogram(data=eigenvalues, dataclass=spectral_config, path=path)


def calc_nn_spacing_dist(spacings: np.ndarray, path: str = "") -> None:
    # Create and save histogram of nearest neighbor spacings
    _create_histogram(data=spacings, dataclass=spacings_config, path=path)


def form_factors_logtimes(dim: int):
    # Create and return logtime array
    return np.logspace(
        sff_config.logtime_min,
        sff_config.logtime_max,
        sff_config.logtime_num,
        base=dim,
    )


def save_form_factors(
    times: np.ndarray, sff: np.ndarray, csff: np.ndarray, path: str = ""
) -> None:
    # Create default path if not provided
    if path == "":
        path = f"res/data/{sff_config.data_filename}"

    # Save form factors data
    if path.endswith(".npz"):
        np.savez_compressed(
            path,
            times=times,
            sff=sff,
            csff=csff,
        )
    elif path.endswith(".csv"):
        np.savetxt(
            path,
            np.column_stack((times, sff, csff)),
            delimiter=",",
            header="Logtime, SFF, cSFF",
            comments="",
        )
    else:
        raise ValueError(f"Unsupported file type. Expected .npz or .csv, got {path}")


def save_form_factors(
    times: np.ndarray, sff: np.ndarray, csff: np.ndarray, file_type: str = ".npz"
) -> None:
    # Create results directory
    data_dir = MonteCarlo._create_output_dir(res_type="data")

    # Save form factors data
    if file_type == ".npz":
        np.savez_compressed(
            os.path.join(data_dir, sff_config.data_filename),
            times=times,
            sff=sff,
            csff=csff,
        )
    elif file_type == ".csv":
        np.savetxt(
            os.path.join(data_dir, sff_config.data_filename),
            np.column_stack((times, sff, csff)),
            delimiter=",",
            header="Logtime, SFF, cSFF",
            comments="",
        )
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def plot_spectral_hist():
    pass


def plot_nn_spacing_dist():
    pass


def plot_form_factors():
    pass


# =============================
# 3. Spectral Statistics Class
# =============================
class SpectralStatistics(MonteCarlo):
    def _create_worker_args(self) -> List[Dict]:
        # Calculate realizations per worker and remainder
        realizs_per_worker, remainder = divmod(self.realizs, self.workers)

        # Initialize realization array
        realizs_array = np.full(self.workers, realizs_per_worker, dtype=int)

        # Distribute remainder realizations
        realizs_array[:remainder] += 1

        # Initialize list of worker arguments
        worker_args = [None for _ in range(self.workers)]

        # Loop over worker argument list
        for i in range(self.workers):
            # Write worker dictionary argument
            worker_args[i] = {
                "ens_args": self._ens_args,
                "sim_args": {"realizs": realizs_array[i]},
            }

        # Return list of worker arguments
        return worker_args

    @staticmethod
    def _worker_func(args: dict) -> np.ndarray:
        # Unpack the arguments
        ens_args = args["ens_args"]
        sim_args = args["sim_args"]

        # Copy ensemble arguments and pop name
        ens_inputs = ens_args.copy()
        ens_inputs.pop("name")

        # Initialize ensemble
        module = import_module(f"rmtpy.ensembles.{ens_args['name']}")
        ENSEMBLE = getattr(module, module.class_name)
        ensemble = ENSEMBLE(**ens_inputs)

        # Unpack simulation arguments
        realizs = sim_args["realizs"]

        # Return eigenvalues
        return ensemble.eigval_samples(realizs=realizs)

    def run(self):
        pass


# =============================
# 4. Main Function
# =============================
def main():
    # Create argument parser
    parser = ArgumentParser(description="Spectral Statistics Monte Carlo")

    # Retrieve Monte Carlo arguments
    mc_args = SpectralStatistics._parse_args(parser)

    # Initialize spectral statistics simulation class
    mc = SpectralStatistics(**mc_args)

    # Run spectral statistics simulation
    mc.run()


# Run the main function
if __name__ == "__main__":
    main()
