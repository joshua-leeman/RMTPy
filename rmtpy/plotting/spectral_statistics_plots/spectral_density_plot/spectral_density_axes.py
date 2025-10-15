# rmtpy/plotting/spectral_statistics_plots/spectral_density_plot/spectral_density_axes.py

# Postponed evaluation of annotations
from __future__ import annotations

# Standard library imports
from dataclasses import dataclass

# Third-party imports
import numpy as np

# Local application imports
from ..._base import PlotAxes


# -------------------------------
# Spectral Density Axes Dataclass
# -------------------------------
@dataclass(repr=False, eq=False, kw_only=True)
class SpectralDensityAxes(PlotAxes):

    # -----------------
    # x-axis Properties
    # -----------------

    # x-axis labels
    xlabel: str = r"$E$"
    unf_xlabel: str = r"$\xi$"

    # x-axis ticks
    xticks: tuple[float, ...] = (-1.0, 0.0, 1.0)  # factor of E0
    unf_xticks: tuple[float, ...] = (-1.0, 0.0, 1.0)  # factor of D/2

    # x-axis minor ticks
    xticks_minor: tuple[float, ...] = (-0.5, 0.5)  # factor of E0
    unf_xticks_minor: tuple[float, ...] = (-1.5, -0.5, 0.5, 1.5)  # factor of D/2

    # x-axis tick labels
    xtick_labels: tuple[str, ...] = (
        r"$-NJ$",
        r"$0$",
        r"$NJ$",
    )
    unf_xtick_labels: tuple[str, ...] = (
        r"$-\frac{1}{2}D$",
        r"$0$",
        r"$\frac{1}{2}D$",
    )

    # -----------------
    # y-axis Properties
    # -----------------

    # y-axis labels
    ylabel: str = r"$\ensavg{\rho(E)}$"
    unf_ylabel: str = r"$\ensavg{\rho(\xi)}$"

    # y-axis ticks
    yticks: tuple[float, ...] = (0.0, 1.0, 2.0)  # factor of 1/pi/(E0)
    unf_yticks: tuple[float, ...] = (0.0, 0.5, 1.0)  # factor of 1/D

    # y-axis minor ticks
    yticks_minor: tuple[float, ...] = (0.0, 1.0, 2.0)  # factor of 1/pi/(E0)
    unf_yticks_minor: tuple[float, ...] = (0.25, 0.75, 1.25)  # factor of 1/D

    # y-axis tick labels
    ytick_labels: tuple[str, ...] = (
        r"$0$",
        r"$\frac{1}{\pi NJ}$",
        r"$\frac{2}{\pi NJ}$",
    )
    unf_ytick_labels: tuple[str, ...] = (
        r"$0$",
        r"$\frac{1}{2}D^{-1}$",
        r"$D^{-1}$",
    )

    # -------------
    # Special Cases
    # -------------

    # Poisson specific y-axis ticks
    poi_yticks: tuple[float, ...] = (
        0.0,
        0.25 * np.pi,
        0.5 * np.pi,
        0.75 * np.pi,
    )  # factor of 1/pi/(E0)
    poi_yticks_minor: tuple[float, ...] = (
        0.125 * np.pi,
        0.375 * np.pi,
        0.625 * np.pi,
        0.875 * np.pi,
    )  # factor of 1/pi/(E0)

    # Poisson specific y-axis tick labels
    poi_ytick_labels: tuple[str, ...] = (
        r"$0$",
        r"$\frac{1}{4NJ}$",
        r"$\frac{1}{2NJ}$",
        r"$\frac{3}{4NJ}$",
    )

    # -------------

    # SYK q=2 specific y-axis ticks
    syk2_yticks: tuple[float, ...] = tuple(range(6))  # factor of 1/pi/(E0)
    syk2_yticks_minor: tuple[float, ...] = tuple(
        x + 0.5 for x in range(6)
    )  # factor of 1/pi/(E0)

    # SYK q=2 specific y-axis tick labels
    syk2_ytick_labels: tuple[str, ...] = (
        r"$0$",
        r"$\frac{1}{\pi NJ}$",
        r"$\frac{2}{\pi NJ}$",
        r"$\frac{3}{\pi NJ}$",
        r"$\frac{4}{\pi NJ}$",
        r"$\frac{5}{\pi NJ}$",
    )

    # SYK q=4 specific y-axis ticks
    syk4_yticks: tuple[float, ...] = tuple(range(3))  # factor of 1/pi/(E0)
    syk4_yticks_minor: tuple[float, ...] = tuple(
        x + 0.5 for x in range(3)
    )  # factor of 1/pi/(E0)

    # SYK q=4 specific y-axis tick labels
    syk4_ytick_labels: tuple[str, ...] = (
        r"$0$",
        r"$\frac{1}{\pi NJ}$",
        r"$\frac{2}{\pi NJ}$",
    )
