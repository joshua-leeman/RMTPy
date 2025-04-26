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
from argparse import ArgumentParser
from multiprocessing import Pool, shared_memory
from textwrap import dedent
from time import time
from typing import Any

# Third-party imports
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import LogLocator, NullLocator
from psutil import virtual_memory
from scipy.special import jn_zeros

# Local application imports
from rmtpy.utils import get_ensemble, _create_plot, _initialize_plot
from rmtpy.simulations._mc import MonteCarlo
from rmtpy.configs.evolve_cdo_config import (
    statistics_config,
    probability_config,
    purity_config,
    entropy_config,
    # expectation_config,
)


# =============================
# 2. Plotting Functions
# =============================
def _plot_cdo_statistics(
    data_path: str,
    dataclass: Any,
    ensemble: Any,
    unfold: bool,
    fig: Figure,
    ax: Axes,
    logy: bool = False,
) -> None:
    """
    Helper function to plot CDO statistics.
    """
    # Set x-axis to logarithmic scale
    ax.set_xscale("log", base=ensemble.dim)
    ax.xaxis.set_major_locator(
        LogLocator(base=ensemble.dim, numticks=dataclass.num_ticks)
    )

    # If logy is True, set y-axis to logarithmic scale
    if logy:
        ax.set_yscale("log", base=ensemble.dim)
        ax.yaxis.set_major_locator(
            LogLocator(base=ensemble.dim, numticks=dataclass.num_ticks)
        )

    # Turn off x- and y-axis minor ticks
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())

    # Store first positive zero of the Bessel function of the first kind
    j_1_1 = jn_zeros(1, 1)[0]

    # Create tick time values
    tick_times = (
        np.logspace(
            start=dataclass.unfolded_logtime_min,
            stop=dataclass.unfolded_logtime_max,
            num=dataclass.num_ticks,
            base=ensemble.dim,
            dtype=np.float64,
        )
        * (2 * np.pi)
        if unfold
        else np.logspace(
            start=dataclass.logtime_min,
            stop=dataclass.logtime_max,
            num=dataclass.num_ticks,
            base=ensemble.dim,
            dtype=np.float64,
        )
        * (2 * j_1_1)
        / (2 * ensemble.N * ensemble.J)
    )

    # Add content to plot based on unfolding
    if not unfold:
        # Create major x-axis grid lines
        ax.vlines(
            tick_times[1:-1],
            ymin=ensemble.dim**-3,
            ymax=ensemble.dim,
            color=dataclass.grid_color,
            linestyle=dataclass.grid_linestyle,
            linewidth=dataclass.grid_linewidth,
            zorder=dataclass.grid_zorder,
        )

    # Set x-axis limits
    ax.set_xlim(tick_times[0], tick_times[-1])

    # Create ticks for x-axis
    ax.set_xticks(tick_times[1:-1])

    # Set y-limits and ticks depending on logy
    if logy:
        # Set y-limits
        ax.set_ylim(
            ensemble.dim**dataclass.logy_min,
            ensemble.dim**dataclass.logy_max,
        )
        # Create ticks for y-axis
        ax.set_yticks([ensemble.dim**i for i in range(-2, 1)])
    else:
        # Set y-limits
        ax.set_ylim(0, 1.5)

        # Create ticks for y-axis
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticks([0.25, 0.75, 1.25], minor=True)

    # Set legend title
    legend_title = rf"{ensemble}" + ("\nunfolded" if unfold else "")

    # Finish plot and save it
    _create_plot(
        dataclass=dataclass,
        data_path=data_path,
        legend_title=legend_title,
        fig=fig,
        ax=ax,
        unfold=unfold,
    )


def plot_probabilities(data_path: str) -> None:
    """
    Plots the cdo revival and an alternative probability from the data file.

    Parameters
    ----------
    data_path : str
        Path to the data file containing histogram data.

    Raises
    ------
    FileNotFoundError
        If the specified path does not exist.
    ValueError
        If the file name does not match the expected name or if the ensemble name is not found in the path.
    """
    # Initialize plot
    ensemble, data, unfold, fig, ax = _initialize_plot(probability_config, data_path)

    # Unpack data
    times = data["times"]
    revival_probabilities = data["revival_probabilities"]
    other_probabilities = data["other_probabilities"]

    # Plot revival probability
    ax.plot(
        times,
        revival_probabilities,
        color=probability_config.revival_color,
        linewidth=probability_config.revival_width,
        alpha=probability_config.revival_alpha,
        zorder=probability_config.revival_zorder,
    )

    # Plot other probability
    ax.plot(
        times,
        other_probabilities,
        color=probability_config.other_color,
        linewidth=probability_config.other_width,
        alpha=probability_config.other_alpha,
        zorder=probability_config.other_zorder,
    )

    # Finish plot
    _plot_cdo_statistics(
        data_path, probability_config, ensemble, unfold, fig, ax, logy=True
    )


def plot_purity(data_path: str) -> None:
    """
    Plots the purity of the CDO from the data file

    Parameters
    ----------
    data_path : str
        Path to the data file containing histogram data.

    Raises
    ------
    FileNotFoundError
        If the specified path does not exist.
    ValueError
        If the file name does not match the expected name or if the ensemble name is not found in the path.
    """
    # Initialize plot
    ensemble, data, unfold, fig, ax = _initialize_plot(purity_config, data_path)

    # Unpack data
    times = data["times"]
    purity = data["purity"]

    # Plot purity
    ax.plot(
        times,
        purity,
        color=purity_config.purity_color,
        linewidth=purity_config.purity_width,
        alpha=purity_config.purity_alpha,
        zorder=purity_config.purity_zorder,
    )

    # Finish plot
    _plot_cdo_statistics(data_path, purity_config, ensemble, unfold, fig, ax, logy=True)


def plot_entropy(data_path: str) -> None:
    """
    Plots the von Neumann entropy of the CDO from the data file

    Parameters
    ----------
    data_path : str
        Path to the data file containing histogram data.

    Raises
    ------
    FileNotFoundError
        If the specified path does not exist.
    ValueError
        If the file name does not match the expected name or if the ensemble name is not found in the path.
    """
    # Initialize plot
    ensemble, data, unfold, fig, ax = _initialize_plot(entropy_config, data_path)

    # Unpack data
    times = data["times"]
    entropy = data["entropy"]

    # Plot entropy
    ax.plot(
        times,
        entropy,
        color=entropy_config.entropy_color,
        linewidth=entropy_config.entropy_width,
        alpha=entropy_config.entropy_alpha,
        zorder=entropy_config.entropy_zorder,
    )

    # Finish plot
    _plot_cdo_statistics(
        data_path, entropy_config, ensemble, unfold, fig, ax, logy=False
    )


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
    run() -> None
        Runs all specified simulations.
    """

    def __init__(
        self,
        ensemble: dict,
        realizs: int,
        workers: int,
        memory: float = virtual_memory().available // 2**30,
        unfold: bool = False,
    ) -> None:
        """
        Initialize the EvolveCDO class.

        Parameters
        ----------
        ensemble : dict
            Random matrix ensemble parameters.
        realizs : int, optional
            Number of realizations (default is 1).
        workers : int, optional
            Number of workers (default is 1).
        memory : float, optional
            Memory allocated for simulation in GiB.
        unfold : bool, optional
            Whether to unfold eigenvalues (default is False).
        """
        # Initialize Monte Carlo simulation class
        super().__init__(ensemble, realizs, workers, memory)

        # Store unfold flag
        self._unfold = unfold

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

        # Add arguments for which simulation(s) to unfold eigenvalues
        parser.add_argument(
            "-unfold",
            "--unfold",
            type=bool,
            default=False,
            help=dedent(
                """
                Specify whether to unfold eigenvalues (True) or not (False).
                """
            ),
        )

        # Send parser to Monte Carlo simulation class and return arguments
        return MonteCarlo._parse_args(parser)

    @staticmethod
    def _realize_evolved_states(args: dict) -> np.ndarray:
        """
        Worker function to generate a sample of evolved states.

        Parameters
        ----------
        args : dict
            Dictionary containing worker id, ensemble and simulation arguments.
        """
        # Unpack arguments
        worker_id = args["worker_id"]
        num_workers = args["num_workers"]
        ens_args = args["ens_args"]
        sim_args = args["sim_args"]
        shm_name = args["shm_name"]

        # Unpack simulation arguments
        realizs = sim_args["realizs"]
        unfold = sim_args["unfold"]

        # Copy ensemble arguments and pop name
        ens_inputs = ens_args.copy()
        ens_inputs.pop("name")

        # Initialize ensemble
        ensemble = get_ensemble(ens_args["name"], **ens_inputs)

        # Store first positive zero of the Bessel function of the first kind
        j_1_1 = jn_zeros(1, 1)[0]

        # Create time array
        times = (
            np.logspace(
                statistics_config.unfolded_logtime_min,
                statistics_config.unfolded_logtime_max,
                statistics_config.num_logtimes,
                base=ensemble.dim,
                dtype=np.float64,
            )
            * (2 * np.pi)
            if unfold
            else np.logspace(
                statistics_config.logtime_min,
                statistics_config.logtime_max,
                statistics_config.num_logtimes,
                base=ensemble.dim,
                dtype=np.float64,
            )
            * (2 * j_1_1)
            / (2 * ensemble.N * ensemble.J)
        )

        # Declare initial state
        initial_state = np.zeros(ensemble.dim, dtype=ensemble.dtype)
        initial_state[0] = 1.0 + 0.0j

        # Calculate number of realizations per worker and remainder
        realizs_per_worker, remainder = divmod(realizs, num_workers)

        # Initialize realizations array
        realizs_array = np.full(num_workers, realizs_per_worker, dtype=np.int64)

        # Distribute remainder realizations to workers
        realizs_array[:remainder] += 1

        # Determine beginning and end indices for given worker_id
        start = np.sum(realizs_array[:worker_id])
        end = start + realizs_array[worker_id]

        # Find shared memory block
        shm = shared_memory.SharedMemory(name=shm_name)

        # View shared memory block as a numpy array
        evolved_states = np.ndarray(
            (statistics_config.num_logtimes, realizs, ensemble.dim),
            dtype=ensemble.dtype,
            buffer=shm.buf,
        )

        # Evolve initial pure state
        ensemble.evolve_pure_state(
            state=initial_state,
            times=times,
            realizs=realizs_array[worker_id],
            unfold=unfold,
            out=evolved_states[:, start:end, :],
        )

        # Close shared memory block
        shm.close()

    @staticmethod
    def _realize_cdo_statistics(args: dict) -> np.ndarray:
        """
        Worker function to generate CDOs and their statistics.

        Parameters
        ----------
        args : dict
            Dictionary containing worker id, ensemble and simulation arguments.
        """
        # Unpack arguments
        worker_id = args["worker_id"]
        num_workers = args["num_workers"]
        ens_args = args["ens_args"]
        sim_args = args["sim_args"]
        shm_name = args["shm_name"]

        # Unpack simulation arguments
        realizs = sim_args["realizs"]
        unfold = sim_args["unfold"]

        # Copy ensemble arguments and pop name
        ens_inputs = ens_args.copy()
        ens_inputs.pop("name")

        # Initialize ensemble
        ensemble = get_ensemble(ens_args["name"], **ens_inputs)

        # Store first positive zero of the Bessel function of the first kind
        j_1_1 = jn_zeros(1, 1)[0]

        # Create time array
        times = (
            np.logspace(
                statistics_config.unfolded_logtime_min,
                statistics_config.unfolded_logtime_max,
                statistics_config.num_logtimes,
                base=ensemble.dim,
                dtype=np.float64,
            )
            * (2 * np.pi)
            if unfold
            else np.logspace(
                statistics_config.logtime_min,
                statistics_config.logtime_max,
                statistics_config.num_logtimes,
                base=ensemble.dim,
                dtype=np.float64,
            )
            * (2 * j_1_1)
            / (2 * ensemble.N * ensemble.J)
        )

        # Array-split times for each worker and delete times
        times_split = np.array_split(times, num_workers)
        del times

        # Grab subset of times for given worker_id
        times = times_split[worker_id]

        # Determine beginning and end indices for given worker_id
        start = int(np.sum([len(t) for t in times_split[:worker_id]]))
        end = start + len(times)

        # Find shared memory block
        shm = shared_memory.SharedMemory(name=shm_name)

        # View shared memory block as a numpy array
        evolved_states = np.ndarray(
            (statistics_config.num_logtimes, realizs, ensemble.dim),
            dtype=ensemble.dtype,
            buffer=shm.buf,
        )

        # Create empty array for CDO statistics
        cdo_statistics = np.empty((len(times), 5), dtype=ensemble.real_dtype)

        # Store times in CDO statistics
        cdo_statistics[:, 0] = times

        # Calculate CDO and its statistics
        for i, t in enumerate(range(start, end)):
            # Calculate CDO
            cdo = ensemble.calculate_cdo(evolved_states[t, :, :])

            # Calculate probabilities
            probabilities = ensemble.cdo_probabilities(cdo)

            # Store revival probability
            cdo_statistics[i, 1] = probabilities[0]

            # Store other probability
            cdo_statistics[i, 2] = probabilities[1]

            # Calculate and store purity
            cdo_statistics[i, 3] = ensemble.cdo_purity(cdo)

            # Calculate and store entropy
            cdo_statistics[i, 4] = ensemble.cdo_entropy(cdo)

            # Delete CDO
            del cdo

        # Close shared memory block
        shm.close()

        # Return CDO statistics
        return cdo_statistics

    def run(self) -> None:
        """
        Run all simulations.
        """
        # Start timer
        start_time = time()

        # Calculate memory required for generating sample of evoled states
        shm_size = (
            np.dtype(self.ensemble.dtype).itemsize
            * self.realizs
            * self.ensemble.dim
            * statistics_config.num_logtimes
        )

        # Create shared memory block for evolved states
        shm = shared_memory.SharedMemory(create=True, size=shm_size)

        # Create set of worker arguments
        worker_args = [
            {
                "worker_id": i,
                "num_workers": self.workers,
                "ens_args": self._ens_args,
                "sim_args": {
                    "realizs": self.realizs,
                    "unfold": self.unfold,
                },
                "shm_name": shm.name,
            }
            for i in range(self.workers)
        ]

        # Realize evolved states in parallel
        with Pool(processes=self.workers) as pool:
            pool.map(
                self._realize_evolved_states,
                worker_args,
            )

        # Realize CDO statistics in parallel
        with Pool(processes=self.workers) as pool:
            cdo_statistics = np.vstack(
                pool.map(
                    self._realize_cdo_statistics,
                    worker_args,
                )
            )

        # Close shared memory block
        shm.close()

        # Unlink shared memory block
        shm.unlink()

        # Create output directory
        output_dir = self._create_output_dir(res_type="data")

        # Create results path based on unfolding
        if self.unfold:
            data_path = os.path.join(
                output_dir, statistics_config.unfolded_data_filename
            )
        else:
            data_path = os.path.join(output_dir, statistics_config.data_filename)

        # Save CDO statistics to file
        np.savez_compressed(
            data_path,
            times=cdo_statistics[:, 0],
            revival_probabilities=cdo_statistics[:, 1],
            other_probabilities=cdo_statistics[:, 2],
            purity=cdo_statistics[:, 3],
            entropy=cdo_statistics[:, 4],
        )

        # Plot revival and other probabilities
        plot_probabilities(data_path)

        # Plot purity
        plot_purity(data_path)

        # Plot entropy
        plot_entropy(data_path)

        # Plot expectation values
        # plot_expectation(data_path)

        # Stop timer and store elapsed time
        elapsed_time = time() - start_time

        # Print elapsed time
        print(f"CDO Evolution completed in {elapsed_time:.2f} seconds.")

    @property
    def calc_memory(self) -> int:
        """
        Memory required for the simulation in bytes.
        """
        return int(
            self.ensemble.matrix_memory
            + 600 * np.sqrt(self.ensemble.matrix_memory)
            + statistics_config.num_logtimes
            / self._workers
            * np.dtype(self.ensemble.dtype).itemsize
            * self._realizs
            * self.ensemble.dim
        )


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
