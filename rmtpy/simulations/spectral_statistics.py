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
from multiprocessing import Pool
from time import time
from typing import Any

# Third-party imports
import numpy as np
from matplotlib.ticker import LogLocator, NullLocator
from scipy.special import jn_zeros

# Local application imports
from rmtpy.utils import get_ensemble, _create_plot, _initialize_plot
from rmtpy.simulations._mc import MonteCarlo
from rmtpy.configs.spectral_statistics_config import (
    sff_config,
    spectral_config,
    spacings_config,
)


# =============================
# 2. Plotting Functions
# =============================
def plot_spectral_hist(data_path: str) -> None:
    """
    Plots the spectral histogram from the given data path.

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
    ensemble, hist_data, unfold, fig, ax = _initialize_plot(spectral_config, data_path)

    # Unpack histogram data
    hist_counts = hist_data["hist_counts"]
    hist_edges = hist_data["hist_edges"]

    # Plot histogram
    ax.hist(
        hist_edges[:-1],
        bins=hist_edges,
        weights=hist_counts,
        color=spectral_config.hist_color,
        alpha=spectral_config.hist_alpha,
    )

    # Create array of levels
    energies = (
        np.linspace(
            -ensemble.dim / 2, ensemble.dim / 2, num=spectral_config.density_num
        )
        if unfold
        else np.linspace(-ensemble.E0, ensemble.E0, num=spectral_config.density_num)
    )

    # Evaluate theoretical average spectral density
    density = (
        np.full_like(energies, 1 / ensemble.dim)
        if unfold
        else np.vectorize(ensemble.spectral_density)(energies)
    )

    # Plot theoretical average spectral density
    ax.plot(
        energies,
        density,
        color=spectral_config.curve_color,
        linewidth=spectral_config.curve_width,
        zorder=spectral_config.curve_zorder,
    )

    # Set limits and ticks based on unfolding
    if unfold:
        # Set x-axis limits
        ax.set_xlim(
            -spectral_config.x_range * ensemble.dim / 2,
            spectral_config.x_range * ensemble.dim / 2,
        )

        # Set y-axis limits
        ax.set_ylim(0, 1.5 / ensemble.dim)

        # Create major ticks for x-axis
        ax.set_xticks((-ensemble.dim / 2, 0, ensemble.dim / 2))

        # Create minor ticks for x-axis
        ax.set_xticks((-ensemble.dim / 4, ensemble.dim / 4), minor=True)

        # Create major ticks for y-axis
        ax.set_yticks((0, 0.5 / ensemble.dim, 1 / ensemble.dim))
    else:
        # Set x-axis limits
        ax.set_xlim(
            -spectral_config.x_range * ensemble.E0,
            spectral_config.x_range * ensemble.E0,
        )

        # Adjust y-axis limits only for Poisson ensemble
        if ensemble.__class__.__name__ == "Poisson":
            ax.set_ylim(0, 1.5 / 2 / ensemble.N / ensemble.J)

        # Create major ticks for x-axis
        ax.set_xticks((-ensemble.E0, 0, ensemble.E0))

        # Create minor ticks for x-axis
        ax.set_xticks((-ensemble.E0 / 2, ensemble.E0 / 2), minor=True)

    # Set legend title
    legend_title = rf"{ensemble}" + ("\nunfolded" if unfold else "")

    # Finish plot and save it
    _create_plot(
        dataclass=spectral_config,
        data_path=data_path,
        legend_title=legend_title,
        fig=fig,
        ax=ax,
        unfold=unfold,
    )


def plot_nn_spacing_hist(data_path: str) -> None:
    """
    Plots the nearest-neighbor level spacing distribution from the given data path.

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
    ensemble, hist_data, unfold, fig, ax = _initialize_plot(spacings_config, data_path)

    # Update simulation configuration with universal class
    spacings_config._set_universal_class(ensemble.universal_class)

    # Unpack histogram data
    hist_counts = hist_data["hist_counts"]
    hist_edges = hist_data["hist_edges"]

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

    # Create major ticks for x-axis
    ax.set_xticks(spacings_config.major_xticks)

    # Create minor ticks for x-axis
    ax.set_xticks(spacings_config.minor_xticks, minor=True)

    # Set legend title
    legend_title = rf"{ensemble}" + ("\nunfolded" if unfold else "")

    # Finish plot and save it
    _create_plot(
        dataclass=spacings_config,
        data_path=data_path,
        legend_title=legend_title,
        fig=fig,
        ax=ax,
        unfold=unfold,
    )


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
    # Initialize plot
    ensemble, form_factors_data, unfold, fig, ax = _initialize_plot(
        sff_config, data_path
    )

    # Update simulation configuration with universal class
    sff_config._set_universal_class(ensemble.universal_class)

    # Unpack form factors data
    times = form_factors_data["times"]
    sff = form_factors_data["sff"]
    csff = form_factors_data["csff"]

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

    # Store first positive zero of the Bessel function of the first kind
    j_1_1 = jn_zeros(1, 1)[0]

    # Create tick time values
    tick_times = (
        np.logspace(
            start=sff_config.unfolded_logtime_min,
            stop=sff_config.unfolded_logtime_max,
            num=sff_config.num_ticks,
            base=ensemble.dim,
            dtype=np.float64,
        )
        * (2 * np.pi)
        if unfold
        else np.logspace(
            start=sff_config.logtime_min,
            stop=sff_config.logtime_max,
            num=sff_config.num_ticks,
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
            color=sff_config.grid_color,
            linestyle=sff_config.grid_linestyle,
            linewidth=sff_config.grid_linewidth,
            zorder=sff_config.grid_zorder,
        )
    else:
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

    # Set x-axis limits
    ax.set_xlim(tick_times[0], tick_times[-1])

    # Create ticks for x-axis
    ax.set_xticks(tick_times[1:-1])

    # Set y-limits
    ax.set_ylim(ensemble.dim**sff_config.logy_min, ensemble.dim**sff_config.logy_max)

    # Create ticks for y-axis
    ax.set_yticks([ensemble.dim**i for i in range(-2, 1)])

    # Set legend title
    legend_title = rf"{ensemble}" + ("\nunfolded" if unfold else "")

    # Finish plot and save it
    _create_plot(
        dataclass=sff_config,
        data_path=data_path,
        legend_title=legend_title,
        fig=fig,
        ax=ax,
        unfold=unfold,
    )


# =============================
# 3. Spectral Statistics Class
# =============================
class SpectralStatistics(MonteCarlo):
    """
    SpectralStatistics class for performing Monte Carlo simulations to obtain spectral statistics of random matrix ensembles.
    Inherits from the MonteCarlo class.

    Methods
    -------
    run_simulation(sim_num, levels=None)
        Run a specific simulation and save the results.
    run()
        Run all specified simulations.
    """

    @staticmethod
    def _worker_func(args: dict) -> np.ndarray:
        """
        Worker function to run the simulation on separate processes.

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
        ensemble = get_ensemble(ens_args["name"], **ens_inputs)

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

        # Create list of worker arguments
        worker_args = [
            {"ens_args": self._ens_args, "sim_args": {"realizs": realizs_array[i]}}
            for i in range(self.workers)
        ]

        # Run workers in parallel
        with Pool(processes=self.workers) as pool:
            eigenvals = np.vstack(pool.map(self._worker_func, worker_args))  # type: ignore

        # Return eigenvalues
        return eigenvals

    def _create_hist(
        self, data: np.ndarray, dataclass: Any, unfold: bool = False
    ) -> None:
        """
        Creates a histogram from the given data and saves it to a file.

        Parameters
        ----------
        data : np.ndarray
            Data to create the histogram from.
        dataclass : Any
            Configuration class containing histogram parameters.
        """
        # Calculate normalized histogram of data
        hist_counts, hist_edges = np.histogram(
            data, bins=dataclass.num_bins, density=True
        )

        # Create output directory
        output_dir = self._create_output_dir(res_type="data")

        # Create results path based on unfolding
        if unfold:
            data_path = os.path.join(output_dir, dataclass.unfolded_data_filename)
        else:
            data_path = os.path.join(output_dir, dataclass.data_filename)

        # Save histogram data
        np.savez_compressed(
            data_path,
            hist_counts=hist_counts,
            hist_edges=hist_edges,
        )

        # Return results path
        return data_path

    def spectral_hist(self, levels: np.ndarray, unfold: bool = False) -> str:
        """
        Create a histogram of the eigenvalue sample.

        Parameters
        ----------
        levels : np.ndarray
            Eigenvalue sample.
        unfold : bool, optional
            Whether to unfold eigenvalues (default is False).

        Returns
        -------
        str
            Path to the saved eigenvalue histogram data file.
        """
        # Create histogram using levels as data
        data_path = self._create_hist(
            data=levels, dataclass=spectral_config, unfold=unfold
        )

        # Return results path
        return data_path

    def nn_spacing_hist(self, levels: np.ndarray, unfold: bool = False) -> str:
        """
        Create a histogram of the nearest-neighbor level spacing sample.

        Parameters
        ----------
        levels : np.ndarray
            Eigenvalue sample.

        Returns
        -------
        str
            Path to the saved nearest-neighbor spacing data file.
        """
        # Calculate nearst neighbor spacings
        spacings = self.ensemble.nn_spacings(levels=levels)

        # If unfolding is not requested, divide spacings by effective mean level spacing
        if not unfold:
            spacings /= np.mean(spacings) / self.ensemble.degen

        # Create histogram using spacings as data
        data_path = self._create_hist(
            data=spacings, dataclass=spacings_config, unfold=unfold
        )

        # Return results path
        return data_path

    def form_factors(self, levels: np.ndarray, unfold: bool = False) -> str:
        """
        Create a plot of the spectral form factors versus time.

        Parameters
        ----------
        levels : np.ndarray
            Eigenvalue sample.
        unfold : bool, optional
            Whether to unfold eigenvalues (default is False).

        Returns
        -------
        str
            Path to the saved form factors data file.
        """
        # Store first positive zero of the Bessel function of the first kind
        j_1_1 = jn_zeros(1, 1)[0]

        # Create logtime array depending on unfolding
        times = (
            np.logspace(
                sff_config.unfolded_logtime_min,
                sff_config.unfolded_logtime_max,
                sff_config.num_logtimes,
                base=self.ensemble.dim,
                dtype=np.float64,
            )
            * (2 * np.pi)
            if unfold
            else np.logspace(
                start=sff_config.logtime_min,
                stop=sff_config.logtime_max,
                num=sff_config.num_logtimes,
                base=self.ensemble.dim,
                dtype=np.float64,
            )
            * (2 * j_1_1)
            / (2 * self.ensemble.N * self.ensemble.J)
        )

        # Allocate memory for form factors
        sff = np.empty_like(times, dtype=np.float64)
        csff = np.empty_like(times, dtype=np.float64)

        # Loop over chunks and evaluate form factors
        for i, time in enumerate(times):
            # Calculate form factors for the current chunk
            sff[i : i + 1], csff[i : i + 1] = self.ensemble.form_factors(
                time=time, levels=levels
            )

        # Create output directory
        output_dir = self._create_output_dir(res_type="data")

        # Create results path based on unfolding
        if unfold:
            data_path = os.path.join(output_dir, sff_config.unfolded_data_filename)
        else:
            data_path = os.path.join(output_dir, sff_config.data_filename)

        # Save form factors data
        np.savez_compressed(
            data_path,
            times=times,
            sff=sff,
            csff=csff,
        )

        # Return results path
        return data_path

    def run(self) -> None:
        """
        Run all specified simulations.
        """
        # Start timer
        start_time = time()

        # Realize eigenvalues
        levels = self._realize_eigvals()

        for unfold in (False, True):
            # Unfold eigenvalues if requested
            if unfold:
                # Unfold eigenvalues
                levels = self.ensemble.unfold(levels)

            # Create histogram of eigenvalues
            data_path = self.spectral_hist(levels=levels, unfold=unfold)
            plot_spectral_hist(data_path=data_path)

            # Create histogram of nearest-neighbor spacings
            data_path = self.nn_spacing_hist(levels=levels, unfold=unfold)
            plot_nn_spacing_hist(data_path=data_path)

            # Create plot of spectral form factors
            data_path = self.form_factors(levels=levels, unfold=unfold)
            plot_form_factors(data_path=data_path)

        # Stop timer and store elapsed time
        elapsed_time = time() - start_time

        # Print elapsed time
        print(f"Spectral statistics completed in {elapsed_time:.2f} seconds.")

    @property
    def calc_memory(self) -> int:
        """
        Memory required for the simulation in bytes.
        """
        return self.ensemble.matrix_memory + 300 * np.sqrt(self.ensemble.matrix_memory)


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
