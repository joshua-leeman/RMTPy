# rmtpy.plotting._plot.py


# =======================================
# 1. Imports
# =======================================
# Standard library imports
from __future__ import annotations
from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

# Local application imports
from rmtpy.utils import ensemble_from_path, configure_matplotlib


# =======================================
# 2. Functions
# =======================================
def _parse_plot_args(parser: ArgumentParser) -> dict[str]:
    """Parse command line arguments for plotting."""
    # Add argument for data path
    parser.add_argument(
        "-f",
        "--data_path",
        type=str,
        required=True,
        help="Path to the data file to be plotted.",
    )

    # Return parsed arguments as a dictionary
    return vars(parser.parse_args())


# =======================================
# 3. Axes Dataclass
# =======================================
@dataclass(repr=False, eq=False, kw_only=True)
class PlotAxes:
    # Axes settings
    axes_width: float = 1.0

    # Axes labels
    xlabel: str = r"$x$"
    ylabel: str = r"$y$"
    fontsize: int = 12

    # Axes ticks
    xticks: Optional[tuple[float, ...]] = None
    yticks: Optional[tuple[float, ...]] = None
    xticks_minor: Optional[tuple[float, ...]] = None
    yticks_minor: Optional[tuple[float, ...]] = None
    tick_length: float = 6.0

    # Axes tick labels
    xticklabels: Optional[tuple[str, ...]] = None
    yticklabels: Optional[tuple[str, ...]] = None
    tick_fontsize: int = 10

    def _set_axes(self, ax: Axes) -> None:
        """Sets the axes properties for the plot."""
        # Set axes width
        for spine in ax.spines.values():
            spine.set_linewidth(self.axes_width)

        # Set x-axis label
        ax.set_xlabel(self.xlabel, fontsize=self.fontsize)

        # Set y-axis label
        ax.set_ylabel(self.ylabel, fontsize=self.fontsize)

        # Set x-ticks
        if self.xticks is not None:
            ax.set_xticks(self.xticks)

        # Set minor x-ticks
        if self.xticks_minor is not None:
            ax.set_xticks(self.xticks_minor, minor=True)

        # Set y-ticks
        if self.yticks is not None:
            ax.set_yticks(self.yticks)

        # Set minor y-ticks
        if self.yticks_minor is not None:
            ax.set_yticks(self.yticks_minor, minor=True)

        # Set tick parameters
        ax.tick_params(
            direction="in",
            top=True,
            bottom=True,
            left=True,
            right=True,
            which="both",
            length=self.tick_length,
        )

        # Set x-tick labels
        if self.xticklabels is not None:
            ax.set_xticklabels(self.xticklabels, fontsize=self.tick_fontsize)
        else:
            # Change x-tick labels' font size
            ax.tick_params(axis="x", labelsize=self.tick_fontsize)

        # Set y-tick labels
        if self.yticklabels is not None:
            ax.set_yticklabels(self.yticklabels, fontsize=self.tick_fontsize)
        else:
            # Change y-tick labels' font size
            ax.tick_params(axis="y", labelsize=self.tick_fontsize)


# =======================================
# 4. Legend Dataclass
# =======================================
@dataclass(repr=False, eq=False, kw_only=True)
class PlotLegend:
    # Legend handles
    handles: Optional[tuple[Any, ...]] = None

    # Legend labels
    labels: Optional[tuple[str, ...]] = None

    # Legend text properties
    fontsize: int = 10
    textalignment: str = "left"

    # Legend title
    title: str = None
    title_fontsize: int = 10

    # Legend location
    loc: str = "best"
    bbox: Optional[tuple[float, float]] = None

    # Legend frame
    frameon: bool = False

    def _set_legend(self, ax: Axes) -> None:
        """Sets the legend properties for the plot."""
        # Set legend if handles and labels are provided
        if self.handles is not None and self.labels is not None:
            ax.legend(
                handles=self.handles,
                labels=self.labels,
                title=self.title,
                loc=self.loc,
                bbox_to_anchor=self.bbox,
                frameon=self.frameon,
                fontsize=self.fontsize,
                title_fontsize=self.title_fontsize,
                alignment=self.textalignment,
            )


# =======================================
# 5. Plot Dataclass
# =======================================
@dataclass(repr=False, eq=False, kw_only=True)
class Plot:
    # Data file name
    data_path: Optional[str] = None

    # Data
    data: Optional[np.ndarray] = None

    # Plot file name
    file_name: Optional[str] = None

    # Axes limits
    xlim: Optional[tuple[float, float]] = None
    ylim: Optional[tuple[float, float]] = None

    # Axes configuration
    axes: PlotAxes = field(default_factory=PlotAxes)

    # Legend configuration
    legend: PlotLegend = field(default_factory=PlotLegend)

    def __post_init__(self):
        # Extract ensemble from data path
        self.ensemble = ensemble_from_path(self.data_path)

        # Load data from file if not provided
        if self.data is None:
            # Convert data path to Path object
            data_path = Path(self.data_path)

            # Check if data path is a file
            if not data_path.exists():
                raise FileNotFoundError(f"Data file {data_path} does not exist.")

            # Load data from file
            self.data = np.load(data_path)

        # Configure matplotlib
        configure_matplotlib()

    def create_figure(self):
        """Creates a figure and axes for plotting."""
        # Create figure and axes
        self.fig, self.ax = plt.subplots()

        # Close figure to avoid displaying it
        plt.close(self.fig)

    def save_plot(self, unfold: bool) -> None:
        # Create plot path from data path
        data_path = Path(self.data_path)
        plot_dir = data_path.parent / "plots"

        # Create unfold prefix if unfolding is enabled
        if unfold:
            prefix = "unf_"
        else:
            prefix = ""

        # Create plot directory if it doesn't exist
        plot_dir.mkdir(parents=True, exist_ok=True)
        if self.file_name is None:
            self.file_name = prefix + data_path.stem + ".png"
        else:
            self.file_name = prefix + self.file_name

        # Create plot path
        plot_path = plot_dir / self.file_name

        # Save plot to file
        self.fig.savefig(plot_path, bbox_inches="tight", dpi=300)

    def set_axes(self) -> None:
        """Sets the axes properties for the plot."""
        # Check if figure and axes are created
        if self.fig is None or self.ax is None:
            raise ValueError("Figure and axes must be created before plotting.")

        # Set axes properties
        self.axes._set_axes(self.ax)

    def set_legend(self) -> None:
        """Sets the legend for the plot."""
        # Check if figure and axes are created
        if self.fig is None or self.ax is None:
            raise ValueError("Figure and axes must be created before plotting.")

        # Set legend properties
        self.legend._set_legend(self.ax)

    def set_plot(self, unfold: bool) -> None:
        """Finishes plot with specified parameters and saves it to a file."""
        # Check if figure and axes are created
        if self.fig is None or self.ax is None:
            raise ValueError("Figure and axes must be created before plotting.")

        # Set x-axis limits
        if self.xlim is not None:
            self.ax.set_xlim(self.xlim)

        # Set y-axis limits
        if self.ylim is not None:
            self.ax.set_ylim(self.ylim)

        # Set axes properties
        self.set_axes()

        # Set legend properties
        self.set_legend()

        # Save plot to file
        self.save_plot(unfold=unfold)
