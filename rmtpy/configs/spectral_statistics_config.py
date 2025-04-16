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
    x_range: float = 1.2  # factor of scale

    # Results parameters
    data_filename: str = "spectrum.npz"
    plot_filename: str = "spectrum.png"

    # Plot settings
    title: str = "Average Spectral Density"
    hist_legend: str = "simulation"
    curve_legend: str = "theory"
    xlabel: str = r"$E$"
    ylabel: str = r"$\langle \rho(E) \rangle$"
    unfolded_xlabel: str = r"$\xi$"
    unfolded_ylabel: str = r"$\langle \rho(\xi) \rangle$"
    hist_color: str = "RoyalBlue"
    curve_color: str = "Black"
    hist_alpha: float = 0.5
    curve_width: float = 2.5
    axes_width: float = 1.0
    hist_zorder: int = 1
    curve_zorder: int = 2


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

    # Results parameters
    data_filename: str = "spacings.npz"
    plot_filename: str = "spacings.png"

    # Plot settings
    title: str = "NNS Distribution"
    hist_legend: str = "simulation"
    curve_legend: str = "surmise"
    xlabel: str = r"$\varepsilon$"
    ylabel: str = r"$\langle \rho(\varepsilon / d) \rangle$"
    unfolded_xlabel: str = r"$s$"
    unfolded_ylabel: str = r"$\langle \rho(s) \rangle$"
    hist_color: str = "Orange"
    curve_color: str = "Black"
    hist_alpha: float = 0.5
    curve_width: float = 2.5
    axes_width: float = 1.0
    hist_zorder: int = 1
    curve_zorder: int = 2


# =============================
# 4. Form Factors Data Class
# =============================
@dataclass
class SpectralFormFactors:
    """
    Data class for spectral form factors simulation parameters.
    """

    # Simulation parameters
    logtime_num: int = 5000
    logtime_min: float = -1.75
    logtime_max: float = 1.00

    # Results parameters
    data_filename: str = "form_factors.npz"
    plot_filename: str = "form_factors.png"

    # Plot settings
    title: str = "Spectral Form Factors"
    sff_legend: str = r"$\textrm{SFF}$"
    csff_legend: str = r"$\textrm{cSFF}$"
    universal_legend: str = "theory"
    xlabel: str = r"$\tau = J t$"
    ylabel: str = r"$K(\tau)$"
    sff_color: str = "Red"
    csff_color: str = "Blue"
    universal_color: str = "Black"
    sff_alpha: float = 1.0
    csff_alpha: float = 1.0
    sff_width: float = 1.0
    csff_width: float = 1.0
    universal_width: float = 1.0
    axes_width: float = 1.0
    sff_zorder: int = 3
    csff_zorder: int = 4
    universal_zorder: int = 5


# =============================
# 5. Instantiations
# =============================
# Instantiate data classes
spectral_config = SpectralHistogram()
spacings_config = NNLevelSpacings()
sff_config = SpectralFormFactors()

# Set matplotlib rcParams for plots
rcParams["text.usetex"] = True
rcParams["font.family"] = "serif"
rcParams["font.serif"] = "Latin Modern Roman"
rcParams["font.size"] = 12
