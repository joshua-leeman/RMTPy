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
from dataclasses import dataclass, field
from typing import Tuple

# Third-party imports
from matplotlib.pyplot import rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# =============================
# 2. Spectral Histogram Data Class
# =============================
@dataclass
class SpectralHistogram:
    """
    Data class for spectral histogram simulation parameters.
    """

    # Simulation parameters
    num_bins: int = 250
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
    has_xticklabels: bool = True
    xticklabels: Tuple[str, ...] = (r"$-\frac{1}{2}NJ$", r"$0$", r"$\frac{1}{2}NJ$")
    has_unfolded_xticklabels: bool = True
    unfolded_xticklabels: Tuple[str, ...] = (
        r"$-\frac{1}{2}D$",
        r"$0$",
        r"$\frac{1}{2}D$",
    )
    has_yticklabels: bool = False
    has_unfolded_yticklabels: bool = True
    unfolded_yticklabels: Tuple[str, ...] = (r"$0$", r"$\frac{1}{2}D$", r"$D$")
    ticklabel_fontsize: int = 10
    tick_length: int = 6

    # Legend settings
    legend_handles: Tuple[Patch, Line2D] = (
        Patch(color=hist_color, alpha=hist_alpha),
        Line2D([0], [0], color=curve_color, linewidth=curve_width),
    )
    legend_labels: Tuple[str, str] = (hist_legend, curve_legend)
    unfolded_legend_handles: Tuple[Patch, Line2D] = legend_handles
    unfolded_legend_labels: Tuple[str, str] = legend_labels
    legend_location: str = "upper right"
    legend_bbox: Tuple[float, float] = (0.99, 0.96)
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

    # Set universal class at initialization
    universal_class: str = field(init=False)

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
    has_xticklabels: bool = True
    xticklabels: Tuple[str, ...] = tuple(
        [r"$d$" if i == 1 else rf"${i}d$" for i in range(1, x_max + 1)]
    )
    has_unfolded_xticklabels: bool = True
    unfolded_xticklabels: Tuple[str, ...] = tuple(
        [rf"${i}.0$" for i in range(1, x_max + 1)]
    )
    has_yticklabels: bool = False
    has_unfolded_yticklabels: bool = False
    major_xticks: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0)
    minor_xticks: Tuple[float, ...] = (0.5, 1.5, 2.5, 3.5)
    ticklabel_fontsize: int = 10
    tick_length: int = 6

    # Legend settings
    legend_handles: Tuple[Patch, Line2D] = (
        Patch(color=hist_color, alpha=hist_alpha),
        Line2D([0], [0], color=curve_color, linewidth=curve_width),
    )
    legend_labels: Tuple[str, str] = (hist_legend, curve_legend)
    unfolded_legend_handles: Tuple[Patch, Line2D] = legend_handles
    unfolded_legend_labels: Tuple[str, str] = legend_labels
    legend_location: str = "upper right"
    legend_bbox: Tuple[float, float] = (0.94, 0.9)
    legend_fontsize: int = 10
    legend_title_fontsize: int = 10
    legend_frameon: bool = False
    legend_textalignment: str = "left"

    # Set attributes depending on universal class
    def _set_universal_class(self, universal_class: str):
        # Store universal class
        self.universal_class = universal_class

        # Set universal class-specific attributes
        self.curve_legend = f"{universal_class} surmise"
        self.legend_labels = (self.hist_legend, self.curve_legend)
        self.unfolded_legend_labels = self.legend_labels


# =============================
# 4. Form Factors Data Class
# =============================
@dataclass
class SpectralFormFactors:
    """
    Data class for spectral form factors simulation parameters
    """

    # Set universal class at initialization
    universal_class: str = field(init=False)

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
    has_xticklabels: bool = True
    xticklabels: Tuple[str, ...] = (r"$1/N$", r"$\sqrt{D} / N$", r"$D / N$")
    has_unfolded_xticklabels: bool = True
    unfolded_xticklabels: Tuple[str, ...] = (
        r"$D^{-1}$",
        r"$D^{-1/2}$",
        r"$1$",
    )
    has_yticklabels: bool = True
    yticklabels: Tuple[str, ...] = (r"$D^{-2}$", r"$D^{-1}$", r"$1$")
    has_unfolded_yticklabels: bool = True
    unfolded_yticklabels: Tuple[str, ...] = yticklabels
    ticklabel_fontsize: int = 10
    tick_length: int = 6

    # Legend settings
    legend_handles: Tuple[Line2D, Line2D] = (
        Line2D(
            [0],
            [0],
            color=sff_color,
            alpha=sff_alpha,
            linewidth=sff_width,
        ),
        Line2D(
            [0],
            [0],
            color=csff_color,
            alpha=csff_alpha,
            linewidth=csff_width,
        ),
    )
    legend_labels: Tuple[str, str] = (sff_legend, csff_legend)
    unfolded_legend_handles: Tuple[Line2D, Line2D, Line2D] = (
        Line2D(
            [0],
            [0],
            color=sff_color,
            alpha=sff_alpha,
            linewidth=sff_width,
        ),
        Line2D(
            [0],
            [0],
            color=csff_color,
            alpha=csff_alpha,
            linewidth=csff_width,
        ),
        Line2D(
            [0],
            [0],
            color=universal_color,
            alpha=universal_alpha,
            linewidth=universal_width,
        ),
    )
    unfolded_legend_labels: Tuple[str, str, str] = (
        sff_legend,
        csff_legend,
        universal_legend,
    )
    legend_location: str = "upper right"
    legend_bbox: Tuple[float, float] = (0.735, 0.95)
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

    # Set attributes depending on universal class
    def _set_universal_class(self, universal_class: str):
        # Store universal class
        self.universal_class = universal_class

        # Set universal class-specific attributes
        self.universal_legend = f"{universal_class} limit"
        self.unfolded_legend_labels = (
            self.sff_legend,
            self.csff_legend,
            self.universal_legend,
        )


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
