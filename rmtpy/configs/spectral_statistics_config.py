# rmtpy.configs.spectral_statistics_config.py
"""
This module contains the data classes related to the spectral statistics simulations.
It is grouped into the following sections:
    1. Imports
    2. Spectral Histogram Data Class
    3. NN Level Spacings Data Class
    4. Form Factors Data Class
    5. Instantiations
"""


# =============================
# 1. Imports
# =============================
# Standard library imports
from dataclasses import dataclass
from typing import Tuple

# Third-party imports
from matplotlib.pyplot import rcParams


# =============================
# 2. Spectral Histogram Data Class
# =============================
@dataclass
class SpectralHistogram:
    """
    Data class for spectral histogram simulation parameters.
    """

    # Simulation parameters
    num_bins: int = 100
    density_num: int = 1000
    x_range: float = 1.2  # factor of J

    # File names
    data_filename: str = "spectrum.npz"
    plot_filename: str = "spectrum.png"
    unfolded_data_filename: str = "unfolded_spectrum.npz"
    unfolded_plot_filename: str = "unfolded_spectrum.png"

    # Histogram graphic parameters
    hist_color: str = "RoyalBlue"
    hist_alpha: float = 0.5
    hist_zorder: int = 1

    # Curve graphic parameters
    curve_color: str = "Black"
    curve_width: float = 2.0
    curve_zorder: int = 2

    # Axes settings
    axes_width: float = 1.0

    # Plot settings
    hist_legend: str = "simulation"
    curve_legend: str = "theory"
    xlabel: str = r"$E$"
    ylabel: str = r"$\langle \rho(E) \rangle$"
    unfolded_xlabel: str = r"$\xi$"
    unfolded_ylabel: str = r"$\langle \rho(\xi) \rangle$"

    # Tick settings
    xticklabels: Tuple[str, ...] = (r"$-\frac{1}{2}NJ$", r"$0$", r"$\frac{1}{2}NJ$")
    unfolded_xticklabels: Tuple[str, ...] = (
        r"$-\frac{1}{2}D$",
        r"$0$",
        r"$\frac{1}{2}D$",
    )
    unfolded_yticklabels: Tuple[str, ...] = (r"$0$", r"$\frac{1}{2}D$", r"$D$")
    ticklabel_fontsize: int = 10
    tick_length: int = 6

    # Legend settings
    legend_location: str = "upper right"
    legend_bbox: Tuple[float, float] = (0.935, 0.96)
    legend_fontsize: int = 10
    legend_title_fontsize: int = 10
    legend_frameon: bool = False
    legend_textalignment: str = "left"


# =============================
# 3. NN Level Spacings Data Class
# =============================
@dataclass
class NNLevelSpacings:
    """
    Data class for nearest neighbor level spacings simulation parameters.
    """

    # Simulation parameters
    num_bins: int = 100
    density_num: int = 1000
    x_max: int = 4

    # File names
    data_filename: str = "spacings.npz"
    plot_filename: str = "spacings.png"
    unfolded_data_filename: str = "unfolded_spacings.npz"
    unfolded_plot_filename: str = "unfolded_spacings.png"

    # Histogram graphic parameters
    hist_color: str = "Orange"
    hist_alpha: float = 0.5
    hist_zorder: int = 1

    # Curve graphic parameters
    curve_color: str = "Black"
    curve_width: float = 2.0
    curve_zorder: int = 2

    # Axes settings
    axes_width: float = 1.0

    # Plot settings
    hist_legend: str = "simulation"
    curve_legend: str = "surmise"
    xlabel: str = r"$\varepsilon$"
    ylabel: str = r"$\langle \rho(\varepsilon) \rangle$"
    unfolded_xlabel: str = r"$s$"
    unfolded_ylabel: str = r"$\langle \rho(s) \rangle$"

    # Tick settings
    ticklabel_fontsize: int = 10
    tick_length: int = 6

    # Legend settings
    legend_location: str = "upper right"
    legend_bbox: Tuple[float, float] = (0.9, 0.9)
    legend_fontsize: int = 10
    legend_title_fontsize: int = 10
    legend_frameon: bool = False
    legend_textalignment: str = "left"


# =============================
# 4. Form Factors Data Class
# =============================
@dataclass
class SpectralFormFactors:
    """
    Data class for spectral form factors simulation parameters
    """

    # Simulation parameters
    num_logtimes: int = 5000
    logtime_min: float = -0.5
    logtime_max: float = 1.5
    unfolded_logtime_min: float = -1.5
    unfolded_logtime_max: float = 0.5

    # Number of tick times
    num_ticks: int = 5

    # File names
    data_filename: str = "form_factors.npz"
    plot_filename: str = "form_factors.png"
    unfolded_data_filename: str = "unfolded_form_factors.npz"
    unfolded_plot_filename: str = "unfolded_form_factors.png"

    # SFF curve parameters
    sff_color: str = "Blue"
    sff_alpha: float = 1.0
    sff_width: float = 1.0
    sff_zorder: int = 2

    # cSFF curve parameters
    csff_color: str = "Red"
    csff_alpha: float = 1.0
    csff_width: float = 1.0
    csff_zorder: int = 2

    # Universal curve parameters
    universal_color: str = "Black"
    universal_alpha: float = 1.0
    universal_width: float = 1.5
    universal_zorder: int = 2

    # Axes settings
    axes_width: float = 1.0

    # y-axis limits
    logy_min: float = -2.2
    logy_max: float = 0.2

    # Plot settings
    sff_legend: str = r"SFF"
    csff_legend: str = r"cSFF"
    universal_legend: str = "universal"
    xlabel: str = r"$Jt$"
    ylabel: str = r"$K(Jt)$"
    unfolded_xlabel: str = r"$\tau$"
    unfolded_ylabel: str = r"$K(\tau)$"

    # Tick settings
    xticklabels: Tuple[str, ...] = (r"$1/N$", r"$\sqrt{D} / N$", r"$D / N$")
    unfolded_xticklabels: Tuple[str, ...] = (
        r"$D^{-1}$",
        r"$D^{-1/2}$",
        r"$1$",
    )
    yticklabels: Tuple[str, ...] = (r"$D^{-2}$", r"$D^{-1}$", r"$1$")
    ticklabel_fontsize: int = 10
    tick_length: int = 6

    # Legend settings
    legend_location: str = "upper right"
    legend_bbox: Tuple[float, float] = (0.735, 0.9)
    legend_fontsize: int = 10
    legend_title_fontsize: int = 10
    legend_frameon: bool = False
    legend_textalignment: str = "left"

    # Grid line settings
    grid_color: str = rcParams["grid.color"]
    grid_linestyle: str = "dotted"
    grid_linewidth: float = rcParams["grid.linewidth"]
    grid_alpha: float = 1.0
    grid_zorder: int = 0


# =============================
# 5. Instantiations
# =============================
# Instantiate data classes
spectral_config = SpectralHistogram()
spacings_config = NNLevelSpacings()
sff_config = SpectralFormFactors()

# Set matplotlib rcParams for plots
rcParams["text.usetex"] = True
rcParams["axes.axisbelow"] = False
rcParams["font.family"] = "serif"
rcParams["font.serif"] = "Latin Modern Roman"
rcParams["font.size"] = 12
