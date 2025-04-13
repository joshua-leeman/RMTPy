# rmtpy.simulations.spectral_statistics.py
"""
This module contains programs for performing Monte Carlo simulations to obtain the spectral statistics of random matrix ensembles.
It is grouped into the following sections:
    1. Imports
    2. Plotting Functions
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
# 2. Plotting Functions
# =============================
def form_factors_logtimes(dim: int):
    # Create and return logtime array
    return np.logspace(
        sff_config.logtime_min,
        sff_config.logtime_max,
        sff_config.logtime_num,
        base=dim,
    )


def save_form_factors(
    times: np.ndarray, sff: np.ndarray, csff: np.ndarray, data_path: str = ""
) -> None:
    # Create default data_path if not provided
    if data_path == "":
        data_path = f"res/data/{sff_config.data_filename}"

    # Save form factors data
    if data_path.endswith(".npz"):
        np.savez_compressed(
            data_path,
            times=times,
            sff=sff,
            csff=csff,
        )
    elif data_path.endswith(".csv"):
        np.savetxt(
            data_path,
            np.column_stack((times, sff, csff)),
            delimiter=",",
            header="Logtime, SFF, cSFF",
            comments="",
        )
    else:
        raise ValueError(
            f"Unsupported file type. Expected .npz or .csv, got {data_path}"
        )


def plot_spectral_hist(
    data_path: str,
    plot_path: str = "",
    titled: bool = False,
    energies: np.ndarray = None,
    density: np.ndarray = None,
) -> None:
    # Load histogram data
    hist_data = np.load(data_path)

    # Unpack histogram data
    hist_counts = hist_data["hist_counts"]
    hist_edges = hist_data["hist_edges"]

    # Create default plot_path if not provided
    if plot_path == "":
        plot_path = f"res/plots/{spectral_config.plot_filename}"

    # Initialize figure and axis
    fig, ax = plt.subplots()

    # Set line widths
    for spine in ax.spines.values():
        spine.set_linewidth(spectral_config.axes_width)

    # Plot histogram
    ax.hist(
        hist_edges[:-1],
        bins=hist_edges,
        weights=hist_counts,
        color=spectral_config.hist_color,
        alpha=spectral_config.hist_alpha,
        zorder=spectral_config.hist_zorder,
    )

    # If energies and density are provided, plot them
    if energies is not None and density is not None:
        density_line = ax.plot(
            energies,
            density,
            color=spectral_config.curve_color,
            linewidth=spectral_config.curve_width,
            zorder=spectral_config.curve_zorder,
        )


def plot_nn_spacing_dist():
    pass


def plot_form_factors():
    pass


# =============================
# 3. Spectral Statistics Class
# =============================
class SpectralStatistics(MonteCarlo):

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

    def _realize_eigvals(self) -> List[Dict]:
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

        # Run workers in parallel
        with Pool(processes=self.workers) as pool:
            eigenvals = np.vstack(pool.map(self._worker_func, worker_args))

        # Return eigenvalues
        return eigenvals

    def _create_histogram(self, data: np.ndarray, dataclass: object) -> None:
        # Calculate bin edges
        min_edge, max_edge = np.min(data), np.max(data)

        # Arrange bin edges
        bins = np.arange(min_edge, max_edge + dataclass.bin_width, dataclass.bin_width)

        # Calculate normalized histogram of data
        hist_counts, hist_edges = np.histogram(data, bins=bins, density=True)

        # Create output directory and store results path
        output_dir = self._create_output_dir(res_type="data")
        res_path = os.path.join(output_dir, sff_config.data_filename)

        # Save histogram data
        if res_path.endswith(".npz"):
            np.savez_compressed(
                res_path,
                hist_counts=hist_counts,
                hist_edges=hist_edges,
            )
        elif res_path.endswith(".csv"):
            np.savetxt(
                res_path,
                np.column_stack((hist_edges[:-1], hist_counts)),
                delimiter=",",
                header="Bin Edges, Counts",
                comments="",
            )
        else:
            raise ValueError(
                f"Unsupported file type. Expected .npz or .csv, got {res_path}"
            )

    def _spectral_histogram(self, levels: np.ndarray) -> None:
        # Create histogram using levels as data
        self._create_histogram(data=levels, dataclass=spectral_config)

    def _nn_spacing_dist(self, levels: np.ndarray) -> None:
        # Calculate nearst neighbor spacings
        spacings = self.ensemble.nn_spacing(levels=levels)

        # Create histogram using spacings as data
        self._create_histogram(data=spacings, dataclass=spacings_config)

    def _form_factors(self, levels: np.ndarray) -> None:
        # Create logtime array
        times = np.logspace(
            sff_config.logtime_min,
            sff_config.logtime_max,
            sff_config.logtime_num,
            base=self.ensemble.dim,
            dtype=np.float64,
        )

        # Allocate memory for form factors
        sff = np.empty_like(times, dtype=np.float64)
        csff = np.empty_like(times, dtype=np.float64)

        # Determine the number of batches based on memory
        num_batches = np.ceil(
            times.size
            * self.realizs
            * self.ensemble.dim
            * np.dtype(np.float64).itemsize
            / self.memory
        ).astype(int)

        # Split logtimes into batches
        batched_times = np.array_split(times, num_batches)

        # Loop over batches and calculate form factors
        index = 0
        for batch in batched_times:
            # Calculate and store form factors
            sff[index : index + batch.size], csff[index : index + batch.size] = (
                self.ensemble.form_factors(times=batch, levels=levels)
            )

            # Increment index
            index += batch.size

        # Create output directory and store results path
        output_dir = self._create_output_dir(res_type="data")
        res_path = os.path.join(output_dir, sff_config.data_filename)

        # Save form factors data
        if res_path.endswith(".npz"):
            np.savez_compressed(
                res_path,
                times=times,
                sff=sff,
                csff=csff,
            )
        elif res_path.endswith(".csv"):
            np.savetxt(
                res_path,
                np.column_stack((times, sff, csff)),
                delimiter=",",
                header="Time, SFF, cSFF",
                comments="",
            )
        else:
            raise ValueError(
                f"Unsupported file type. Expected .npz or .csv, got {res_path}"
            )

    def run_spectral_histogram(self, unfold: bool = False) -> None:
        # Start timer
        start_time = time()

        # Realize eigenvalues
        levels = self._realize_eigvals()

        # Unfold eigenvalues if specified
        if unfold:
            levels = self.ensemble.unfold(levels=levels)

        # Calculate spectral histogram
        self._spectral_histogram(levels=levels)

        # Stop timer and store elapsed time
        elapsed_time = time() - start_time

        # Print elapsed time
        print(f"Spectral histogram calculated in {elapsed_time:.2f} seconds.")

    def run_nn_spacing_dist(self, unfold: bool = True) -> None:
        pass

    def run_form_factors(self, unfold: bool = True) -> None:
        pass

    def run_all(self) -> None:
        pass


# =============================
# 4. Main Function
# =============================
def main() -> None:
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
