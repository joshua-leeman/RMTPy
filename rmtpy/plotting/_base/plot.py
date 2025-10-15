# rmtpy/plotting/base/plot.py

# Postponed evaluation of annotations
from __future__ import annotations

# Standard library imports
import inspect
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

# Third-party imports
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Local application imports
from .axes import PlotAxes
from .legend import PlotLegend
from ...data import Data
from ...ensembles import converter


# -------------------------
# Monte Carlo Plot Registry
# -------------------------
PLOT_REGISTRY: dict[str, type[Plot]] = {}


# ------------------------
# Matplotlib configuration
# ------------------------
def configure_matplotlib() -> None:
    """Configure matplotlib settings for plots."""

    # Set matplotlib rcParams for plots
    rcParams["axes.axisbelow"] = False
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = "Latin Modern Roman"

    # Try to use LaTeX for rendering
    try:
        rcParams["text.usetex"] = True
        rcParams["text.latex.preamble"] = "\n".join(
            [
                r"\usepackage{amsmath}",
                r"\newcommand{\ensavg}[1]{\langle\hspace{-0.7ex}\langle #1 \rangle\hspace{-0.7ex}\rangle}",
            ]
        )
    except:
        pass


# ----------------------------
# Plot From Data File Function
# ----------------------------
def plot_data(data_path: str | Path) -> None:
    """Plot data from a specified data file."""

    # Ensure data_path is a Path object
    data_path = Path(data_path)

    # Create Plot instance from data file
    plot = converter.structure(data_path, Plot)

    # Alias parent directory of data file
    out_dir = data_path.parent

    # Plot data and save to parent directory
    plot.plot(path=out_dir)


# ---------------
# Plot Base Class
# ---------------
@dataclass(repr=False, eq=False, kw_only=True)
class Plot(ABC):

    # Plot data
    data: Data

    # x-axis limits
    xlim: tuple[float, float] | None = None

    # y-axis limits
    ylim: tuple[float, float] | None = None

    # Plot axes
    axes: PlotAxes = field(default_factory=PlotAxes)

    # Plot legend
    legend: PlotLegend = field(default_factory=PlotLegend)

    # Plot dots per inch
    dpi: int = 300

    @classmethod
    def __init_subclass__(cls) -> None:
        """Register concrete subclasses in the plot registry."""

        # Include only concrete classes in registry
        if not inspect.isabstract(cls):
            # Convert data class name from CamelCase to snake_case
            plot_key = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", cls.__name__)
            plot_key = plot_key.lower()

            # Replace '_plot' suffix with '_data'
            plot_key = plot_key.replace("_plot", "_data")

            # Normalize class name to registry key format
            PLOT_REGISTRY[plot_key] = cls

    def __post_init__(self) -> None:
        """Post-initialization processing."""

        # Configure matplotlib settings
        configure_matplotlib()

        # Set file name to data file name
        self.file_name = self.data.file_name

    def create_figure(self) -> None:
        """Create a figure and axes for the plot."""

        # Create figure and axis
        self.fig, self.ax = plt.subplots()

        # Close figure to avoid displaying it
        plt.close(self.fig)

    def finish_plot(self, path: str | Path) -> None:
        """Finish and save the plot to path."""

        # Check if figure and axis exist
        if not hasattr(self, "fig") or not hasattr(self, "ax"):
            raise AttributeError(
                "Figure and axis not created. Call create_figure() first."
            )

        # Set x-axis limits
        if self.xlim is not None:
            self.ax.set_xlim(self.xlim)

        # Set y-axis limits
        if self.ylim is not None:
            self.ax.set_ylim(self.ylim)

        # Configure axes
        self.axes.configure(ax=self.ax)

        # Configure legend
        self.legend.configure(ax=self.ax)

        # Ensure path is a Path object
        path = Path(path)

        # Create plot directory if it doesn't exist
        path.mkdir(parents=True, exist_ok=True)

        # Save plot to file
        self.fig.savefig(path / self.data.file_name, dpi=self.dpi, bbox_inches="tight")

    @abstractmethod
    def plot(self, path: str | Path) -> None:
        """Create and save the plot to the specified path."""

        # Abstract method to be implemented by subclasses
        pass
