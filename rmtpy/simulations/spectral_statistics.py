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
from ast import literal_eval
from importlib import import_module
from multiprocessing import Pool
from textwrap import dedent
from time import time
from typing import Dict, List

# Third-party imports
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
from psutil import virtual_memory

# Local application imports
from rmtpy.simulations._mc import MonteCarlo
from rmtpy.configs.spectral_statistics_config import (
    spectral_config,
    spacings_config,
    sff_config,
)

# Load environment variables
load_dotenv()

# Store project path
project_path = os.getenv("PROJECT_PATH")


# =============================
# 2. Plotting Functions
# =============================
def _ensemble_from_path(path: str, file_name: str) -> dict:
    # Check if path exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    # Check if path is named correctly
    if os.path.basename(path) != file_name:
        raise ValueError(f"File name must be '{file_name}'")

    # Initialize metadata dictionary
    metadata = {}

    # Read file path and extract metadata
    for datum in path.split("/"):
        if "=" in datum:
            split_datum = datum.split("=")
            metadata[split_datum[0]] = literal_eval(split_datum[1])

    # Retrieve list of valid ensembles
    ensemble_list = [
        file.rstrip(".py")
        for file in os.listdir(f"{project_path}/rmtpy/ensembles")
        if file.endswith(".py") and not file.startswith("_")
    ]

    # Grabs ensemble name from path
    try:
        ensemble_name = next(
            datum for datum in path.split("/") if datum in ensemble_list
        )
    except StopIteration:
        raise ValueError(f"Ensemble name not found in path: {path}")

    # Copy metadata as ensemble inputs and pop realizations
    ens_inputs = metadata.copy()
    ens_inputs.pop("realizs")

    # Import ensemble module and class
    module = import_module(f"rmtpy.ensembles.{ensemble_name}")
    ENSEMBLE = getattr(module, module.class_name)

    # Return initialized ensemble
    return ENSEMBLE(**ens_inputs)


def plot_spectral_hist(data_path: str) -> None:
    # Reads results path and extracts ensemble
    ensemble = _ensemble_from_path(data_path, sff_config.data_filename)

    # Load histogram data from file
    hist_data = np.load(data_path)

    # Unpack histogram data
    hist_counts = hist_data["hist_counts"]
    hist_edges = hist_data["hist_edges"]

    # Create figure and axis
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
        alpha=spectral_config.hist_alpha,
    )

    # Create array of energy values
    energies = np.linspace(
        -ensemble.scale, ensemble.scale, num=spectral_config.density_num
    )

    # Evaluate theoretical average spectral density
    density = np.vectorize(ensemble.mean_density)(energies)

    # Plot theoretical average spectral density
    ax.plot(
        energies,
        density,
        color=spectral_config.curve_color,
        linewidth=spectral_config.curve_width,
        zorder=spectral_config.curve_zorder,
    )

    # Set axis labels and limits
    ax.set_xlabel(spectral_config.xlabel)
    ax.set_ylabel(spectral_config.ylabel)
    ax.set_xlim(
        -spectral_config.x_range * ensemble.scale,
        spectral_config.x_range * ensemble.scale,
    )

    # Set tick markrs all around and inward
    ax.tick_params(
        direction="in",
        top=True,
        bottom=True,
        left=True,
        right=True,
    )

    # Create plot path from data path
    split_data_path = data_path.split("/")
    split_data_path[-1] = spectral_config.plot_filename
    plot_path = os.path.join(*split_data_path)

    # Save plot to file
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")

    # Close plot
    plt.close(fig)


def plot_nn_spacing_dist():
    pass


def plot_form_factors():
    pass


# =============================
# 3. Spectral Statistics Class
# =============================
class SpectralStatistics(MonteCarlo):
    def __init__(
        self,
        ensemble: dict,
        realizs: int = 1,
        workers: int = 1,
        memory: int = virtual_memory().total // 2**30,
        run: List[int] = [1, 2, 3],
        unfold: List[int] = [2, 3],
    ) -> None:

        # Validate unfold is a subset of run
        if not set(unfold).issubset(set(run)):
            raise ValueError("Unfold must be a subset of run.")

        # Initialize Monte Carlo simulation
        super().__init__(ensemble, realizs, workers, memory)

        # Store which simulations to run
        self.do_spectral_hist = 1 in run
        self.do_nn_spacing_dist = 2 in run
        self.do_form_factors = 3 in run

        # Store which simulations to unfold
        self.unfold_spectral_hist = 1 in unfold
        self.unfold_nn_spacing_dist = 2 in unfold
        self.unfold_form_factors = 3 in unfold

    @staticmethod
    def _parse_args(parser: ArgumentParser) -> dict:
        # Add arguments for which simulation(s) to run
        parser.add_argument(
            "--run",
            nargs="+",
            choices=[1, 2, 3],
            default=[1, 2, 3],
            help=dedent(
                """
                Specify which simulation(s) to run:
                    1: Spectral Histogram
                    2: NN-Level Spacings
                    3: Spectral Form Factors
                """
            ),
        )

        # Add arguments for which simulation(s) to unfold eigenvalues
        parser.add_argument(
            "--unfold",
            nargs="+",
            choices=[1, 2, 3],
            default=[2, 3],
            help=dedent(
                """
                Specify which simulation(s) to unfold eigenvalues (must be subset of --run):
                    1: Spectral Histogram
                    2: NN-Level Spacings
                    3: Spectral Form Factors
                """
            ),
        )

        # Send parser to Monte Carlo simulation class and return arguments
        return MonteCarlo._parse_args(parser)

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

    def _create_hist(self, data: np.ndarray, dataclass: object) -> None:
        # Calculate bin edges
        min_edge, max_edge = np.min(data), np.max(data)

        # Arrange bin edges
        bins = np.arange(min_edge, max_edge + dataclass.bin_width, dataclass.bin_width)

        # Calculate normalized histogram of data
        hist_counts, hist_edges = np.histogram(data, bins=bins, density=True)

        # Create output directory and store results path
        output_dir = self._create_output_dir(res_type="data")
        results_path = os.path.join(output_dir, sff_config.data_filename)

        # Save histogram data
        np.savez_compressed(
            results_path,
            hist_counts=hist_counts,
            hist_edges=hist_edges,
        )

    def _spectral_hist(self, levels: np.ndarray) -> None:
        # Create histogram using levels as data
        self._create_hist(data=levels, dataclass=spectral_config)

    def _nn_spacing_dist(self, levels: np.ndarray) -> None:
        # Calculate nearst neighbor spacings
        spacings = self.ensemble.nn_spacing(levels=levels)

        # Create histogram using spacings as data
        self._create_hist(data=spacings, dataclass=spacings_config)

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
        results_path = os.path.join(output_dir, sff_config.data_filename)

        # Save form factors data
        np.savez_compressed(
            results_path,
            times=times,
            sff=sff,
            csff=csff,
        )

    def run_spectral_hist(self, plot: bool = True) -> None:
        # Start timer
        start_time = time()

        # Realize eigenvalues
        levels = self._realize_eigvals()

        # Unfold eigenvalues if specified
        if self.unfold_spectral_hist:
            levels = self.ensemble.unfold(levels=levels)

        # Calculate spectral histogram
        self._spectral_hist(levels=levels)

        # Plot spectral histogram if specified
        if plot:
            # Determine output directory
            output_dir = self._create_output_dir(res_type="data")

            # Run plot function

        # Stop timer and store elapsed time
        elapsed_time = time() - start_time

        # Print elapsed time
        print(f"Spectral histogram calculated in {elapsed_time:.2f} seconds.")

    def run_nn_spacing_dist(self, plot: bool = True) -> None:
        pass

    def run_form_factors(self, plot: bool = True) -> None:
        pass

    def run(self) -> None:
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
