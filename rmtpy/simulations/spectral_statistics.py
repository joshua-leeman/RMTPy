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
from pathlib import Path
from textwrap import dedent
from time import time
from typing import Dict, List

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullLocator
from psutil import virtual_memory

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
def _ensemble_from_path(path: str, file_name: str) -> dict:
    """
    Determines the ensemble from the given path of data file.

    Parameters
    ----------
    path : str
        Path to the data file.
    file_name : str
        Name of the data file.

    Returns
    -------
    object
        Initialized ensemble object.

    Raises
    ------
    FileNotFoundError
        If the specified path does not exist.
    ValueError
        If the file name does not match the expected name or if the ensemble name is not found in the path.
    """
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
        for file in os.listdir(f"rmtpy/ensembles")
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
    """
    Plots the spectral histogram from the given data path.

    Parameters
    ----------
    data_path : str
        Path to the data file containing histogram data.
    """
    # Reads results path and extracts ensemble
    ensemble = _ensemble_from_path(data_path, spectral_config.data_filename)

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
    )

    # Create array of energy values
    energies = np.linspace(
        -ensemble.scale, ensemble.scale, num=spectral_config.density_num
    )

    # Evaluate theoretical average spectral density
    density = np.vectorize(ensemble.spectral_density)(energies)

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
    data_path = Path(data_path)
    plot_dir = data_path.parent.parent / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / spectral_config.plot_filename

    # Save plot to file
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")

    # Close plot
    plt.close(fig)


def plot_nn_spacing_dist(data_path: str) -> None:
    """
    Plots the nearest-neighbor spacing distribution from the given data path.

    Parameters
    ----------
    data_path : str
        Path to the data file containing histogram data.
    """
    # Reads results path and extracts ensemble
    ensemble = _ensemble_from_path(data_path, spacings_config.data_filename)

    # Load histrogram data from file
    hist_data = np.load(data_path)

    # Unpack histogram data
    hist_counts = hist_data["hist_counts"]
    hist_edges = hist_data["hist_edges"]

    # Create figure and axis
    fig, ax = plt.subplots()

    # Set line widths
    for spine in ax.spines.values():
        spine.set_linewidth(spacings_config.axes_width)

    # Plot histogram
    ax.hist(
        hist_edges[:-1],
        bins=hist_edges,
        weights=hist_counts,
        color=spacings_config.hist_color,
        alpha=spacings_config.hist_alpha,
        zorder=spacings_config.hist_zorder,
    )

    # Create array of spacings values
    spacings = np.linspace(0, spacings_config.x_max, num=spacings_config.density_num)

    # Calculate Wigner surmise distribution
    surmise = ensemble.wigner_surmise(spacings)

    # Plot Wigner surmise distribution
    ax.plot(
        spacings,
        surmise,
        color=spacings_config.curve_color,
        linewidth=spacings_config.curve_width,
        zorder=spacings_config.curve_zorder,
    )

    # Set axis labels and limits
    ax.set_xlabel(spacings_config.xlabel)
    ax.set_ylabel(spacings_config.ylabel)
    ax.set_xlim(0, spacings_config.x_max)

    # Set tick marks all around and inward
    ax.tick_params(
        direction="in",
        top=True,
        bottom=True,
        left=True,
        right=True,
    )

    # Create plot path from data path
    data_path = Path(data_path)
    plot_dir = data_path.parent.parent / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / spacings_config.plot_filename

    # Save plot to file
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")

    # Close plot
    plt.close(fig)


def plot_form_factors(data_path: str) -> None:
    """
    Plots the spectral form factors from the given data path.

    Parameters
    ----------
    data_path : str
        Path to the data file containing form factors data.

    Raises
    ------
    FileNotFoundError
        If the specified path does not exist.
    ValueError
        If the file name does not match the expected name or if the ensemble name is not found in the path.
    """
    # Reads results path and extracts ensemble
    ensemble = _ensemble_from_path(data_path, sff_config.data_filename)

    # Load form factors data from file
    form_factors_data = np.load(data_path)

    # Unpack form factors data
    times = form_factors_data["times"]
    sff = form_factors_data["sff"]
    csff = form_factors_data["csff"]

    # Create figure and axis
    fig, ax = plt.subplots()

    # Set line widths
    for spine in ax.spines.values():
        spine.set_linewidth(sff_config.axes_width)

    # Set x- and y-scales to logarithmic
    ax.set_xscale("log", base=ensemble.dim)
    ax.set_yscale("log", base=ensemble.dim)

    # Turn off x- and y-axis minor ticks
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())

    # Limit number of major ticks on x- and y-axis
    ax.xaxis.set_major_locator(LogLocator(base=ensemble.dim, numticks=6))
    ax.yaxis.set_major_locator(LogLocator(base=ensemble.dim, numticks=6))

    # Plot spectral form factor
    ax.plot(
        times,
        sff,
        color=sff_config.sff_color,
        linewidth=sff_config.sff_width,
        alpha=sff_config.sff_alpha,
        zorder=sff_config.sff_zorder,
    )

    # Plot connected spectral form factor
    ax.plot(
        times,
        csff,
        color=sff_config.csff_color,
        linewidth=sff_config.csff_width,
        alpha=sff_config.csff_alpha,
        zorder=sff_config.csff_zorder,
    )

    # Calculate universal connected spectral form factor
    universal_csff = np.vectorize(ensemble.universal_csff)(times)

    # Plot universal connected spectral form factor
    ax.plot(
        times,
        universal_csff,
        color=sff_config.universal_color,
        linewidth=sff_config.universal_width,
        zorder=sff_config.universal_zorder,
    )

    # Set axis labels and limits
    ax.set_xlabel(sff_config.xlabel)
    ax.set_ylabel(sff_config.ylabel)
    ax.set_xlim(
        ensemble.dim**sff_config.logtime_min,
        ensemble.dim**sff_config.logtime_max,
    )
    ax.set_ylim(0.1 * ensemble.dim ** (-2), 10)

    # Create tick labels for x-axis
    ax.set_xticks(
        [
            ensemble.dim**i
            for i in range(sff_config.logtime_min, sff_config.logtime_max + 1)
        ]
    )
    ax.set_xticklabels(
        [
            rf"$D^{{{i+1}}}$" if i not in [-1, 0] else r"$1$" if i == -1 else r"$D$"
            for i in range(sff_config.logtime_min, sff_config.logtime_max + 1)
        ]
    )

    # Create ticks for y-axis
    ax.set_yticks([ensemble.dim**i for i in range(-2, 1)])
    ax.set_yticklabels([(rf"$D^{{{i}}}$" if i != 0 else r"$1$") for i in range(-2, 1)])

    # Set tick marks all around and inward
    ax.tick_params(
        direction="in",
        top=True,
        bottom=True,
        left=True,
        right=True,
    )

    # Create plot path from data path
    data_path = Path(data_path)
    plot_dir = data_path.parent.parent / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / sff_config.plot_filename

    # Save plot to file
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")

    # Close plot
    plt.close(fig)


# =============================
# 3. Spectral Statistics Class
# =============================
class SpectralStatistics(MonteCarlo):
    def __init__(
        self,
        ensemble: dict,
        realizations: int = 1,
        workers: int = 1,
        memory: int = virtual_memory().total // 2**30,
        run: List[int] = [1, 2, 3],
        unfold: List[int] = [],
    ) -> None:

        # Validate unfold is a subset of run
        if not set(unfold).issubset(set(run)):
            raise ValueError("Unfold must be a subset of run.")

        # Initialize Monte Carlo simulation
        super().__init__(ensemble, realizations, workers, memory)

        # Store job arguments in dictionary
        self._job = {
            1: {
                "do": 1 in run,
                "unfold": 1 in unfold,
                "func": self._spectral_hist,
                "plot": plot_spectral_hist,
                "file": spectral_config.data_filename,
            },
            2: {
                "do": 2 in run,
                "unfold": 2 in unfold,
                "func": self._nn_spacing_dist,
                "plot": plot_nn_spacing_dist,
                "file": spacings_config.data_filename,
            },
            3: {
                "do": 3 in run,
                "unfold": 3 in unfold,
                "func": self._form_factors,
                "plot": plot_form_factors,
                "file": sff_config.data_filename,
            },
        }

    @staticmethod
    def _parse_args(parser: ArgumentParser) -> dict:
        # Add arguments for which simulation(s) to run
        parser.add_argument(
            "--run",
            nargs="+",
            type=int,
            choices=[1, 2, 3],
            default=[1, 2, 3],
            help=dedent(
                """
                Specify which simulation(s) to run:
                    (1) Spectral Histogram
                    (2) NN-Level Spacings
                    (3) Spectral Form Factors
                """
            ),
        )

        # Add arguments for which simulation(s) to unfold eigenvalues
        parser.add_argument(
            "--unfold",
            nargs="+",
            type=int,
            choices=[2, 3],
            default=[],
            help=dedent(
                """
                Specify which simulation(s) to unfold eigenvalues (must be subset of --run and cannot be 1):
                    (2) NN-Level Spacings
                    (3) Spectral Form Factors
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
        return ensemble.eigval_sample(realizs=realizs)

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
        results_path = os.path.join(output_dir, dataclass.data_filename)

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
        spacings = self.ensemble.nn_spacings(levels=levels)

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

    def _run_simulation(
        self, sim_num: int, levels: np.ndarray = None, unfold: bool = False
    ) -> None:
        # Store simulation information
        sim_info = self._job[sim_num]

        # Realize eigenvalues if not provided
        if levels is None:
            levels = self._realize_eigvals()
            # Unfold eigenvalues if specified
            if unfold:
                levels = self.ensemble.unfold(levels)

        # Run simulation functionm
        sim_info["func"](levels)

        # Determine output directory
        output_dir = self._create_output_dir(res_type="data")

        # Run plot function
        sim_info["plot"](os.path.join(output_dir, sim_info["file"]))

    def run_spectral_hist(self, unfold: bool = False) -> None:
        # Run spectral histogram simulation
        self._run_simulation(1, unfold=unfold)

    def run_nn_spacing_dist(self, unfold: bool = False) -> None:
        # Run nearest neighbor spacing distribution simulation
        self._run_simulation(2, unfold=unfold)

    def run_form_factors(self, unfold: bool = False) -> None:
        # Run spectral form factors simulation
        self._run_simulation(3, unfold=unfold)

    def run(self) -> None:
        # Start timer
        start_time = time()

        # Realize eigenvalues
        levels = self._realize_eigvals()

        # Run each specified simulation that does not require unfolding
        for sim_num in self._job:
            if self._job[sim_num]["do"] and not self._job[sim_num]["unfold"]:
                self._run_simulation(sim_num, levels)

        # Unfold eigenvalues if specified
        if any(self._job[sim_num]["unfold"] for sim_num in self._job):
            levels = self.ensemble.unfold(levels)

        # Run each specified simulation that requires unfolding
        for sim_num in self._job:
            if self._job[sim_num]["do"] and self._job[sim_num]["unfold"]:
                self._run_simulation(sim_num, levels)

        # Stop timer and store elapsed time
        elapsed_time = time() - start_time

        # Print elapsed time
        print(f"Spectral statistics completed in {elapsed_time:.2f} seconds.")


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
