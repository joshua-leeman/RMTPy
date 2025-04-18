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
from typing import List

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
def _ensemble_from_path(path: str, file_name: str) -> object:
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


def plot_spectral_hist(data_path: str, unfold: bool = False) -> None:
    """
    Plots the spectral histogram from the given data path.

    Parameters
    ----------
    data_path : str
        Path to the data file containing histogram data.
    unfold : bool, optional
        Whether to unfold eigenvalues (default is False).

    Raises
    ------
    FileNotFoundError
        If the specified path does not exist.
    ValueError
        If the file name does not match the expected name or if the ensemble name is not found in the path.
    """
    # Reads results path and extracts ensemble
    if not unfold:
        ensemble = _ensemble_from_path(data_path, spectral_config.data_filename)
    else:
        ensemble = _ensemble_from_path(
            data_path, spectral_config.unfolded_data_filename
        )

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

    # Create plot based on whether unfolding has occurred
    if not unfold:
        # Create array of energy values
        energies = np.linspace(
            -ensemble.E0, ensemble.E0, num=spectral_config.density_num
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
            -spectral_config.x_range * ensemble.E0,
            spectral_config.x_range * ensemble.E0,
        )

        # Create ticks and tick labels for x-axis
        ax.set_xticks((-ensemble.E0, 0, ensemble.E0))
        ax.set_xticklabels(
            spectral_config.xticklabels, fontsize=spectral_config.ticklabel_fontsize
        )

        # Create minor ticks for x-axis
        ax.set_xticks((-ensemble.E0 / 2, ensemble.E0 / 2), minor=True)

        # Change font size of y-axis tick labels
        ax.tick_params(axis="y", labelsize=spectral_config.ticklabel_fontsize)
    else:
        # Create array of levels
        energies = np.linspace(
            -ensemble.dim / 2, ensemble.dim / 2, num=spectral_config.density_num
        )

        # Evaluate theoretical average spectral density
        density = np.full_like(energies, 1 / ensemble.dim)

        # Plot theoretical average spectral density
        ax.plot(
            energies,
            density,
            color=spectral_config.curve_color,
            linewidth=spectral_config.curve_width,
            zorder=spectral_config.curve_zorder,
        )

        # Set axis labels and limits
        ax.set_xlabel(spectral_config.unfolded_xlabel)
        ax.set_ylabel(spectral_config.unfolded_ylabel)
        ax.set_xlim(
            -spectral_config.x_range * ensemble.dim / 2,
            spectral_config.x_range * ensemble.dim / 2,
        )

        # Create tick labels for x-axis
        ax.set_xticks((-ensemble.dim / 2, 0, ensemble.dim / 2))
        ax.set_xticklabels(
            spectral_config.unfolded_xticklabels,
            fontsize=spectral_config.ticklabel_fontsize,
        )

        # Create minor ticks for x-axis
        ax.set_xticks((-ensemble.dim / 4, ensemble.dim / 4), minor=True)

        # Create tick labels for y-axis
        ax.set_yticks((0, 0.5 / ensemble.dim, 1 / ensemble.dim, 1.5 / ensemble.dim))
        ax.set_yticklabels(
            spectral_config.unfolded_yticklabels,
            fontsize=spectral_config.ticklabel_fontsize,
        )

        # Set minor ticks for y-axis
        ax.set_yticks((1 / (2 * ensemble.dim), 3 / (2 * ensemble.dim)), minor=True)

    # Set tick markrs all around and inward
    ax.tick_params(
        direction="in",
        top=True,
        bottom=True,
        left=True,
        right=True,
        which="both",
        length=spectral_config.tick_length,
    )

    # Store plot file name
    if not unfold:
        plot_file = spectral_config.plot_filename
    else:
        plot_file = spectral_config.unfolded_plot_filename

    # Create plot path from data path
    data_path = Path(data_path)
    plot_dir = data_path.parent.parent / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / plot_file

    # Save plot to file
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")

    # Close plot
    plt.close(fig)


def plot_nn_spacing_dist(data_path: str, unfold: bool = False) -> None:
    """
    Plots the nearest-neighbor level spacing distribution from the given data path.

    Parameters
    ----------
    data_path : str
        Path to the data file containing histogram data.
    unfold : bool, optional
        Whether to unfold eigenvalues (default is False).

    Raises
    ------
    FileNotFoundError
        If the specified path does not exist.
    ValueError
        If the file name does not match the expected name or if the ensemble name is not found in the path.
    """
    # Reads results path and extracts ensemble
    if not unfold:
        ensemble = _ensemble_from_path(data_path, spacings_config.data_filename)
    else:
        ensemble = _ensemble_from_path(
            data_path, spacings_config.unfolded_data_filename
        )

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

    # Set x-limits
    ax.set_xlim(0, spacings_config.x_max)

    # Create labels based on whether unfolding has occurred
    if not unfold:
        # Set axis labels and limits
        ax.set_xlabel(spacings_config.xlabel)
        ax.set_ylabel(spacings_config.ylabel)

    else:
        # Set axis labels
        ax.set_xlabel(spacings_config.unfolded_xlabel)
        ax.set_ylabel(spacings_config.unfolded_ylabel)

    # Create tick labels for x-axis
    ax.set_xticks(range(1, spacings_config.x_max + 1))
    ax.set_xticklabels(
        [rf"${i}.0$" for i in range(1, spacings_config.x_max + 1)],
        fontsize=spacings_config.ticklabel_fontsize,
    )

    # Create minor ticks for x-axis
    ax.set_xticks([0.5, 1.5, 2.5, 3.5], minor=True)

    # Set tick marks all around and inward
    ax.tick_params(
        direction="in",
        top=True,
        bottom=True,
        left=True,
        right=True,
        which="both",
        length=spacings_config.tick_length,
    )

    # Change font size of y-axis tick labels
    ax.tick_params(axis="y", labelsize=spacings_config.ticklabel_fontsize)

    # Store plot file name
    if not unfold:
        plot_file = spacings_config.plot_filename
    else:
        plot_file = spacings_config.unfolded_plot_filename

    # Create plot path from data path
    data_path = Path(data_path)
    plot_dir = data_path.parent.parent / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / plot_file

    # Save plot to file
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")

    # Close plot
    plt.close(fig)


def plot_form_factors(data_path: str, unfold: bool = False) -> None:
    """
    Plots the spectral form factors from the given data path.

    Parameters
    ----------
    data_path : str
        Path to the data file containing form factors data.
    unfold : bool, optional
        Whether to unfold eigenvalues (default is False).

    Raises
    ------
    FileNotFoundError
        If the specified path does not exist.
    ValueError
        If the file name does not match the expected name or if the ensemble name is not found in the path.
    """
    # Reads results path and extracts ensemble
    if not unfold:
        ensemble = _ensemble_from_path(data_path, sff_config.data_filename)
    else:
        ensemble = _ensemble_from_path(data_path, sff_config.unfolded_data_filename)

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
    ax.xaxis.set_major_locator(
        LogLocator(base=ensemble.dim, numticks=sff_config.num_ticks)
    )
    ax.yaxis.set_major_locator(
        LogLocator(base=ensemble.dim, numticks=sff_config.num_ticks)
    )

    # Plot spectral form factor
    (sff_line,) = ax.plot(
        times,
        sff,
        color=sff_config.sff_color,
        linewidth=sff_config.sff_width,
        alpha=sff_config.sff_alpha,
        zorder=sff_config.sff_zorder,
    )

    # Plot connected spectral form factor
    (csff_line,) = ax.plot(
        times,
        csff,
        color=sff_config.csff_color,
        linewidth=sff_config.csff_width,
        alpha=sff_config.csff_alpha,
        zorder=sff_config.csff_zorder,
    )

    # Create plot based on whether unfolding has occurred
    if not unfold:
        # Create tick time values
        tick_times = np.logspace(
            start=sff_config.logtime_min,
            stop=sff_config.logtime_max,
            num=sff_config.num_ticks,
            base=ensemble.dim,
            dtype=np.float64,
        )

        # Normalize tick_times by total spectrum width
        tick_times /= ensemble.N * ensemble.J

        # Set axis labels and limits
        ax.set_xlabel(sff_config.xlabel)
        ax.set_ylabel(sff_config.ylabel)
        ax.set_xlim(tick_times[0], tick_times[-1])

        # Create tick labels for x-axis
        ax.set_xticks(tick_times)
        ax.set_xticklabels(
            sff_config.xticklabels, fontsize=sff_config.ticklabel_fontsize
        )

        # Create legend
        legend = ax.legend(
            handles=[sff_line, csff_line],
            labels=[sff_config.sff_legend, sff_config.csff_legend],
            title=rf"{repr(ensemble)}",
            loc=sff_config.legend_location,
            bbox_to_anchor=sff_config.legend_bbox,
            fontsize=sff_config.legend_fontsize,
            title_fontsize=sff_config.legend_title_fontsize,
            frameon=sff_config.legend_frameon,
        )
        legend._legend_box.align = sff_config.legend_textalignment

        # Set major grid lines only on x-axis
        ax.vlines(
            tick_times,
            ymin=ensemble.dim**-3,
            ymax=ensemble.dim,
            color=sff_config.grid_color,
            linestyle=sff_config.grid_linestyle,
            linewidth=sff_config.grid_linewidth,
            zorder=sff_config.grid_zorder,
        )

    else:
        # Calculate universal connected spectral form factor
        universal_csff = np.vectorize(ensemble.universal_csff)(times)

        # Plot universal connected spectral form factor
        (univ_line,) = ax.plot(
            times,
            universal_csff,
            color=sff_config.universal_color,
            linewidth=sff_config.universal_width,
            zorder=sff_config.universal_zorder,
        )

        # Create tick time values
        tick_times = np.logspace(
            sff_config.unfolded_logtime_min,
            sff_config.unfolded_logtime_max,
            sff_config.num_ticks,
            base=ensemble.dim,
            dtype=np.float64,
        )

        # Set axis labels and limits
        ax.set_xlabel(sff_config.unfolded_xlabel)
        ax.set_ylabel(sff_config.unfolded_ylabel)
        ax.set_xlim(tick_times[0], tick_times[-1])

        # Create tick labels for x-axis
        ax.set_xticks(tick_times)
        ax.set_xticklabels(
            sff_config.unfolded_xticklabels,
            fontsize=sff_config.ticklabel_fontsize,
        )

    # Set y-limits
    ax.set_ylim(ensemble.dim**sff_config.logy_min, ensemble.dim**sff_config.logy_max)

    # Create ticks for y-axis
    ax.set_yticks([ensemble.dim**i for i in range(-2, 1)])
    ax.set_yticklabels(sff_config.yticklabels, fontsize=sff_config.ticklabel_fontsize)

    # Set tick marks all around and inward
    ax.tick_params(
        direction="in",
        top=True,
        bottom=True,
        left=True,
        right=True,
        length=sff_config.tick_length,
        width=sff_config.axes_width,
    )

    # Store plot file name
    if not unfold:
        plot_file = sff_config.plot_filename
    else:
        plot_file = sff_config.unfolded_plot_filename

    # Create plot path from data path
    data_path = Path(data_path)
    plot_dir = data_path.parent.parent / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / plot_file

    # Save plot to file
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")

    # Close plot
    plt.close(fig)


# =============================
# 3. Spectral Statistics Class
# =============================
class SpectralStatistics(MonteCarlo):
    """
    SpectralStatistics class for performing Monte Carlo simulations to obtain spectral statistics of random matrix ensembles.
    Inherits from the MonteCarlo class.

    Methods
    -------
    run_spectral_hist(unfold: bool = False)
        Run the spectral histogram simulation.
    run_nn_spacing_dist(unfold: bool = False)
        Run the nearest-neighbor level spacing distribution simulation.
    run_form_factors(unfold: bool = False)
        Run the spectral form factors simulation.
    run()
        Run all specified simulations.
    """

    def __init__(
        self,
        ensemble: dict,
        realizations: int = 1,
        workers: int = 1,
        memory: int = virtual_memory().total // 2**30,
        run: List[int] = [],
        unfold: List[int] = [],
    ) -> None:
        """
        Initialize the SpectralStatistics simulation class.

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
        run : list of int, optional
            List of simulations to run (default is [1, 2, 3]).
        unfold : list of int, optional
            List of simulations to unfold eigenvalues (default is empty list).

        Raises
        ------
        ValueError
            If unfold is not a subset of run or if 1 is included in unfold.
        """

        # Validate unfold is a subset of run
        if not set(unfold).issubset(set(run)):
            raise ValueError("Unfold must be a subset of run.")

        # Initialize Monte Carlo simulation
        super().__init__(ensemble, realizations, workers, memory)

        # If run is empty, denote all flag and set run to all simulations
        if not run:
            self._all = True
            run = [1, 2, 3]
        else:
            self._all = False

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
        """
        Parse command line arguments for the SpectralStatistics simulation class.

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
            "--run",
            nargs="+",
            type=int,
            choices=[1, 2, 3],
            default=[],
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
        """
        Worker function to run the simulation on a separate processes.

        Parameters
        ----------
        args : dict
            Dictionary containing ensemble and simulation arguments.

        Returns
        -------
        np.ndarray
            Eigenvalue sample from the simulation.
        """
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

    def _realize_eigvals(self) -> np.ndarray:
        """
        Divides the number of realizations among workers and runs the simulation in parallel.

        Returns
        -------
        np.ndarray
            Eigenvalue sample from the simulation.
        """
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

    def _create_hist(
        self, data: np.ndarray, dataclass: object, unfold: bool = False
    ) -> None:
        """
        Creates a histogram from the given data and saves it to a file.

        Parameters
        ----------
        data : np.ndarray
            Data to create the histogram from.
        dataclass : object
            Configuration class containing histogram parameters.
        """
        # Calculate normalized histogram of data
        hist_counts, hist_edges = np.histogram(
            data, bins=dataclass.num_bins, density=True
        )

        # Create output directory and store results path
        output_dir = self._create_output_dir(res_type="data")
        results_path = os.path.join(
            output_dir, f"{unfold * 'unfolded_'}{dataclass.data_filename}"
        )

        # Save histogram data
        np.savez_compressed(
            results_path,
            hist_counts=hist_counts,
            hist_edges=hist_edges,
        )

    def _spectral_hist(self, levels: np.ndarray, unfold: bool = False) -> None:
        """
        Create a histogram of the eigenvalue sample.

        Parameters
        ----------
        levels : np.ndarray
            Eigenvalue sample.
        unfold : bool, optional
            Whether to unfold eigenvalues (default is False).
        """
        # Create histogram using levels as data
        self._create_hist(data=levels, dataclass=spectral_config, unfold=unfold)

    def _nn_spacing_dist(self, levels: np.ndarray, unfold: bool = False) -> None:
        """
        Create a histogram of the nearest-neighbor level spacing sample.

        Parameters
        ----------
        levels : np.ndarray
            Eigenvalue sample.
        """
        # Calculate nearst neighbor spacings
        spacings = self.ensemble.nn_spacings(levels=levels)

        # If unfolding is not requested, divide spacings by mean level spacing
        if not unfold:
            spacings /= np.mean(spacings)

        # Create histogram using spacings as data
        self._create_hist(data=spacings, dataclass=spacings_config, unfold=unfold)

    def _form_factors(self, levels: np.ndarray, unfold: bool = False) -> None:
        """
        Create a plot of the spectral form factors versus time.

        Parameters
        ----------
        levels : np.ndarray
            Eigenvalue sample.
        unfold : bool, optional
            Whether to unfold eigenvalues (default is False).
        """
        # Create logtime array depending on unfolding
        if not unfold:
            # Create logtime array based on ensemble parameters
            times = np.logspace(
                start=sff_config.logtime_min,
                stop=sff_config.logtime_max,
                num=sff_config.num_logtimes,
                base=self.ensemble.dim,
                dtype=np.float64,
            )

            # Create tick time values
            tick_times = np.logspace(
                start=sff_config.logtime_min,
                stop=sff_config.logtime_max,
                num=sff_config.num_ticks,
                base=self.ensemble.dim,
                dtype=np.float64,
            )

            # Normalize times and tick_times by total spectrum width
            times /= self.ensemble.N * self.ensemble.J
            tick_times /= self.ensemble.N * self.ensemble.J

            # Append tick_times to times and sort
            times = np.append(times, tick_times)
            times = np.unique(times)
        else:
            # Create logtime array based on ensemble parameters
            times = np.logspace(
                sff_config.unfolded_logtime_min,
                sff_config.unfolded_logtime_max,
                sff_config.num_logtimes,
                base=self.ensemble.dim,
                dtype=np.float64,
            )

            # Create tick time values
            tick_times = np.logspace(
                sff_config.unfolded_logtime_min,
                sff_config.unfolded_logtime_max,
                sff_config.num_ticks,
                base=self.ensemble.dim,
                dtype=np.float64,
            )

            # Append tick_times to times and sort
            times = np.append(times, tick_times)
            times = np.unique(times)

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
        results_path = os.path.join(
            output_dir, f"{unfold * 'unfolded_'}{sff_config.data_filename}"
        )

        # Save form factors data
        np.savez_compressed(
            results_path,
            times=times,
            sff=sff,
            csff=csff,
        )

    def _run_simulation(self, sim_num: int, levels: np.ndarray = None) -> None:
        """
        Run a simulation and save the results.

        Parameters
        ----------
        sim_num : int
            Simulation number.
        levels : np.ndarray, optional
            Eigenvalue sample (default is None).

        Raises
        ------
        ValueError
            If the simulation number is not valid.
        """
        # Store simulation information
        sim_info = self._job[sim_num]

        # Realize eigenvalues if not provided
        if levels is None:
            levels = self._realize_eigvals()
            # Unfold eigenvalues if specified
            if sim_info["unfold"]:
                levels = self.ensemble.unfold(levels)

        # Run simulation functionm
        sim_info["func"](levels, unfold=sim_info["unfold"])

        # Determine output directory
        output_dir = self._create_output_dir(res_type="data")

        # Run plot function
        sim_info["plot"](
            os.path.join(
                output_dir, f"{sim_info['unfold'] * 'unfolded_'}{sim_info['file']}"
            ),
            unfold=sim_info["unfold"],
        )

    def run_spectral_hist(self) -> None:
        """
        Run the spectral histogram simulation.
        """
        # Run spectral histogram simulation
        self._run_simulation(1)

    def run_nn_spacing_dist(self) -> None:
        """
        Run the nearest-neighbor level spacing distribution simulation.
        """
        # Run nearest neighbor spacing distribution simulation
        self._run_simulation(2)

    def run_form_factors(self) -> None:
        """
        Run the spectral form factors simulation.
        """
        # Run spectral form factors simulation
        self._run_simulation(3)

    def run(self) -> None:
        """
        Run all specified simulations.
        """
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
        elif self._all:
            levels = self.ensemble.unfold(levels)
            for sim_num in self._job:
                self._job[sim_num]["unfold"] = True

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
    """
    Main function to run the Spectral Statistics Monte Carlo simulation from the command line.
    """
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
