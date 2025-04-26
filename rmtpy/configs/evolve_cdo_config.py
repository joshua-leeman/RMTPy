# rmtpy.configs.evolve_cdo_config.py
"""
This module contains the data classes related to the CDO evolution simulation.
It is grouped into the following sections:
    1. Imports
    2. Statistics Class
    3. Probabilities Class
    4. Purity Class
    5. Entropy Class
    6. Expectation Value Class
    7. Instantiations
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
# 2. Statistics Class
# =============================
@dataclass
class Statistics:
    """
    Data class for CDO evolution statistics simulation parameters.
    """

    # Set universal class at initialization
    universal_class: str = field(init=False)

    # Simulation parameters
    num_logtimes: int = 1000
    logtime_min: float = -0.5
    logtime_max: float = 1.5
    unfolded_logtime_min: float = -1.5
    unfolded_logtime_max: float = 0.5

    # Number of tick times
    num_tick_times: int = 5

    # File names
    data_filename: str = "cdo_evolution.npz"
    unfolded_data_filename: str = "unfolded_cdo_evolution.npz"


# =============================
# 3. Probabilities Class
# =============================
@dataclass
class Probabilities:
    """
    Data class for CDO evolution probabilities simulation parameters.
    """

    # Set universal class at initialization
    universal_class: str = field(init=False)

    # Simulation parameters
    num_logtimes: int = 1000
    logtime_min: float = -0.5
    logtime_max: float = 1.5
    unfolded_logtime_min: float = -1.5
    unfolded_logtime_max: float = 0.5

    # Number of tick times
    num_ticks: int = 5

    # File names
    data_filename: str = "cdo_evolution.npz"
    unfolded_data_filename: str = "unfolded_cdo_evolution.npz"

    # Plot file names
    plot_filename: str = "probabilities.png"
    unfolded_plot_filename: str = "unfolded_probabilities.png"

    # Revival probability parameters
    revival_color: str = "Blue"
    revival_alpha: float = 1.0
    revival_width: float = 1.0
    revival_zorder: int = 2

    # Other probability parameters
    other_color: str = "Red"
    other_alpha: float = 1.0
    other_width: float = 1.0
    other_zorder: int = 1

    # Axes settings
    axes_width: float = 1.0

    # y-axis limits
    logy_min: float = -2.2
    logy_max: float = 0.2

    # Plot settings
    revival_legend: str = r"$p_0$"
    other_legend: str = r"$p_k\ (k \neq 0)$"
    xlabel: str = r"$Jt / j_{1,1}$"
    ylabel: str = r"$p(t)$"
    unfolded_xlabel: str = r"$\tau$"
    unfolded_ylabel: str = r"$p(\tau)$"

    # Tick settings
    has_xticklabels: bool = True
    xticklabels: Tuple[str, ...] = (
        r"$1 / N$",
        r"$\sqrt{D} / N$",
        r"$D / N$",
    )
    has_unfolded_xticklabels: bool = True
    unfolded_xticklabels: Tuple[str, ...] = (
        r"$2\pi / D$",
        r"$2\pi / \sqrt{D}$",
        r"$2\pi$",
    )
    has_yticklabels: bool = True
    yticklabels: Tuple[str, ...] = (r"$D^{-2}$", r"$D^{-1}$", r"$1$")
    has_unfolded_yticklabels: bool = True
    unfolded_yticklabels: Tuple[str, ...] = yticklabels
    ticklabel_fontsize: int = 9
    tick_length: int = 6

    # Legend settings
    legend_handles: Tuple[Line2D, Line2D] = (
        Line2D(
            [0],
            [0],
            color=revival_color,
            alpha=revival_alpha,
            linewidth=revival_width,
        ),
        Line2D(
            [0],
            [0],
            color=other_color,
            alpha=other_alpha,
            linewidth=other_width,
        ),
    )
    legend_labels: Tuple[str, str] = (revival_legend, other_legend)
    unfolded_legend_handles: Tuple[Line2D, Line2D, Line2D] = (
        Line2D(
            [0],
            [0],
            color=revival_color,
            alpha=revival_alpha,
            linewidth=revival_width,
        ),
        Line2D(
            [0],
            [0],
            color=other_color,
            alpha=other_alpha,
            linewidth=other_width,
        ),
    )
    unfolded_legend_labels: Tuple[str, str] = (
        revival_legend,
        other_legend,
    )
    legend_location: str = "upper right"
    legend_bbox: Tuple[float, float] = (0.76, 0.95)
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
            self.revival_legend,
            self.other_legend,
            self.universal_legend,
        )


# =============================
# 4. Purity Class
# =============================
@dataclass
class Purity:
    """
    Data class for CDO evolution purity simulation parameters.
    """

    # Set universal class at initialization
    universal_class: str = field(init=False)

    # Simulation parameters
    num_logtimes: int = 1000
    logtime_min: float = -0.5
    logtime_max: float = 1.5
    unfolded_logtime_min: float = -1.5
    unfolded_logtime_max: float = 0.5

    # Number of tick times
    num_ticks: int = 5

    # File names
    data_filename: str = "cdo_evolution.npz"
    unfolded_data_filename: str = "unfolded_cdo_evolution.npz"

    # Plot file names
    plot_filename: str = "purity.png"
    unfolded_plot_filename: str = "unfolded_purity.png"

    # Purity parameters
    purity_color: str = "Red"
    purity_alpha: float = 1.0
    purity_width: float = 1.0
    purity_zorder: int = 2

    # Axes settings
    axes_width: float = 1.0

    # y-axis limits
    logy_min: float = -2.2
    logy_max: float = 0.2

    # Plot settings
    purity_legend: str = r"$\gamma$"
    xlabel: str = r"$Jt / j_{1,1}$"
    ylabel: str = r"$\gamma(t)$"
    unfolded_xlabel: str = r"$\tau$"
    unfolded_ylabel: str = r"$\gamma(\tau)$"

    # Tick settings
    has_xticklabels: bool = True
    xticklabels: Tuple[str, ...] = (
        r"$1 / N$",
        r"$\sqrt{D} / N$",
        r"$D / N$",
    )
    has_unfolded_xticklabels: bool = True
    unfolded_xticklabels: Tuple[str, ...] = (
        r"$2\pi / D$",
        r"$2\pi / \sqrt{D}$",
        r"$2\pi$",
    )
    has_yticklabels: bool = True
    yticklabels: Tuple[str, ...] = (r"$D^{-2}$", r"$D^{-1}$", r"$1$")
    has_unfolded_yticklabels: bool = True
    unfolded_yticklabels: Tuple[str, ...] = yticklabels
    ticklabel_fontsize: int = 9
    tick_length: int = 6

    # Legend settings
    legend_handles: Tuple[Line2D, Line2D] = (
        Line2D(
            [0],
            [0],
            color=purity_color,
            alpha=purity_alpha,
            linewidth=purity_width,
        ),
    )
    legend_labels: Tuple[str,] = (purity_legend,)
    unfolded_legend_handles: Tuple[Line2D, Line2D, Line2D] = (
        Line2D(
            [0],
            [0],
            color=purity_color,
            alpha=purity_alpha,
            linewidth=purity_width,
        ),
    )
    unfolded_legend_labels: Tuple[str,] = (purity_legend,)
    legend_location: str = "upper right"
    legend_bbox: Tuple[float, float] = (0.76, 0.95)
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
            self.purity_legend,
            self.universal_legend,
        )


# =============================
# 5. Entropy Class
# =============================
@dataclass
class Entropy:
    """
    Data class for CDO evolution entropy simulation parameters.
    """

    # Set universal class at initialization
    universal_class: str = field(init=False)

    # Simulation parameters
    num_logtimes: int = 1000
    logtime_min: float = -0.5
    logtime_max: float = 1.5
    unfolded_logtime_min: float = -1.5
    unfolded_logtime_max: float = 0.5

    # Number of tick times
    num_ticks: int = 5

    # File names
    data_filename: str = "cdo_evolution.npz"
    unfolded_data_filename: str = "unfolded_cdo_evolution.npz"

    # Plot file names
    plot_filename: str = "entropy.png"
    unfolded_plot_filename: str = "unfolded_entropy.png"

    # Purity parameters
    entropy_color: str = "Red"
    entropy_alpha: float = 1.0
    entropy_width: float = 1.0
    entropy_zorder: int = 2

    # Axes settings
    axes_width: float = 1.0

    # y-axis limits
    logy_min: float = -2.2
    logy_max: float = 0.2

    # Plot settings
    entropy_legend: str = r"$S$"
    xlabel: str = r"$Jt / j_{1,1}$"
    ylabel: str = r"$S(t) / \log D$"
    unfolded_xlabel: str = r"$\tau$"
    unfolded_ylabel: str = r"$S(\tau) / \log D$"

    # Tick settings
    has_xticklabels: bool = True
    xticklabels: Tuple[str, ...] = (
        r"$1 / N$",
        r"$\sqrt{D} / N$",
        r"$D / N$",
    )
    has_unfolded_xticklabels: bool = True
    unfolded_xticklabels: Tuple[str, ...] = (
        r"$2\pi / D$",
        r"$2\pi / \sqrt{D}$",
        r"$2\pi$",
    )
    has_yticklabels: bool = True
    yticklabels: Tuple[str, ...] = (r"$0$", r"$0.5$", r"$1.0$")
    has_unfolded_yticklabels: bool = True
    unfolded_yticklabels: Tuple[str, ...] = yticklabels
    ticklabel_fontsize: int = 9
    tick_length: int = 6

    # Legend settings
    legend_handles: Tuple[Line2D, Line2D] = (
        Line2D(
            [0],
            [0],
            color=entropy_color,
            alpha=entropy_alpha,
            linewidth=entropy_width,
        ),
    )
    legend_labels: Tuple[str,] = (entropy_legend,)
    unfolded_legend_handles: Tuple[Line2D, Line2D, Line2D] = (
        Line2D(
            [0],
            [0],
            color=entropy_color,
            alpha=entropy_alpha,
            linewidth=entropy_width,
        ),
    )
    unfolded_legend_labels: Tuple[str,] = (entropy_legend,)
    legend_location: str = "upper right"
    legend_bbox: Tuple[float, float] = (0.76, 0.95)
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
            self.entropy_legend,
            self.universal_legend,
        )


# =============================
# 7. Instantiations
# =============================
# Instantiate data classes
statistics_config = Statistics()
probability_config = Probabilities()
purity_config = Purity()
entropy_config = Entropy()

# Set matplotlib rcParams for plots
rcParams["text.usetex"] = True
rcParams["axes.axisbelow"] = False
rcParams["font.family"] = "serif"
rcParams["font.serif"] = "Latin Modern Roman"
rcParams["font.size"] = 12
