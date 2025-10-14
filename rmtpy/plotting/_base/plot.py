# rmtpy/plotting/base/plot.py

# Postponed evaluation of annotations
from __future__ import annotations

# Standard library imports
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

    def set_plot(self, path: str | Path) -> None:
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
