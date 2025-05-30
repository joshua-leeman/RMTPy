# rmtpy.plotting.cdo_evolution.entropy_plot.py


# =======================================
# 1. Imports
# =======================================
# Standard library imports
from __future__ import annotations
from dataclasses import dataclass, field

# Third-party imports
import numpy as np
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullLocator
from scipy.special import jn_zeros

# Local application imports
from rmtpy.plotting._plot import Plot, PlotAxes, PlotLegend


# =======================================
# 2. Axes Dataclass
# =======================================
@dataclass(repr=False, eq=False, kw_only=True)
class EntropyPlotAxes(PlotAxes):
    # x-axis labels
    xlabel: str = r"$Jt / j_{1,1}$"
    unf_xlabel: str = r"$\tau$"

    # y-axis labels
    ylabel: str = r"$S(t)$"
    unf_ylabel: str = r"$S(\tau)$"

    # x-tick labels
    xticklabels: tuple[str, ...] = (r"$1/N$", r"$\sqrt{D}/N$", r"$D/N$")
    unf_xticklabels: tuple[str, ...] = (r"$2\pi / D$", r"$2\pi / \sqrt{D}$", r"$2\pi$")

    # y-tick labels
    yticklabels: tuple[str, ...] = (r"$0$", r"$\frac{1}{2} \ln D$", r"$\ln D$")


# =======================================
# 3. Legend Dataclass
# =======================================
@dataclass(repr=False, eq=False, kw_only=True)
class EntropyLegend(PlotLegend):
    # Legend location
    loc: str = "upper right"
    bbox: tuple[float, float] = (0.75, 0.95)


# =======================================
# 4. Entropy Plot Dataclass
# =======================================
@dataclass(repr=False, eq=False, kw_only=True)
class EntropyPlot(Plot):
    # Unfolded data flag
    unfold: bool = False

    # Plot file name
    file_name: str = "entropy_plot"

    # Entropy curve parameters
    e_color: str = "Red"
    e_alpha: float = 1.0
    e_width: float = 1.0
    e_zorder: int = 2
    e_legend: str = r"$\gamma$"

    # Grid line parameters
    grid_color: str = rcParams["grid.color"]
    grid_linestyle: str = "dotted"
    grid_linewidth: float = rcParams["grid.linewidth"]
    grid_alpha: float = 1.0
    grid_zorder: int = 0

    # Legend handles and labels
    handles: tuple[Patch, Line2D] = (
        Line2D([0], [0], color=e_color, alpha=e_alpha, linewidth=e_width),
    )
    labels: tuple[str, str] = (e_legend,)

    # Unfolded legend handles and labels
    unf_handles: tuple[Patch, Line2D, Line2D] = (
        Line2D([0], [0], color=e_color, alpha=e_alpha, linewidth=e_width),
    )
    unf_labels: tuple[str, str, str] = (e_legend,)

    # Axes configuration
    axes: EntropyPlotAxes = field(default_factory=EntropyPlotAxes)

    def __post_init__(self):
        # Initialize base class
        super(EntropyPlot, self).__post_init__()

        # Switch legend handles and labels for unfolded data
        if self.unfold:
            self.handles = self.unf_handles
            self.labels = self.unf_labels

        # Set legend properties
        self.legend = EntropyLegend(handles=self.handles, labels=self.labels)

        # Set derived attributes
        self.set_derived_attributes()

    def set_derived_attributes(self) -> None:
        """Sets derived attributes for the plot."""
        # Create short notations for ensemble attributes
        dim = self.ensemble.dim
        E0 = self.ensemble.E0

        # Store first positive zero of 1st Bessel function
        j_1_1 = jn_zeros(1, 1)[0]

        # Set y-limits
        if self.ylim is None:
            self.ylim = (0, 1.25)

        # Set y-ticks
        if self.axes.yticks is None:
            self.axes.yticks = (0, 0.5, 1.0)

        # Set minor y-ticks
        if self.axes.yticks_minor is None:
            self.axes.yticks_minor = (0.25, 0.75)

        # Set default legend title
        if self.legend.title is None:
            self.legend.title = self.ensemble._to_latex()

        # Configure plot based on unfolding
        if self.unfold:
            # Replace attributes for unfolded data
            self.axes.xlabel = self.axes.unf_xlabel
            self.axes.ylabel = self.axes.unf_ylabel
            self.axes.xticklabels = self.axes.unf_xticklabels

            # Append legend title with "unfolded"
            self.legend.title += "\nunfolded"

            # Set x-ticks for unfolded data
            self.tick_times = np.logspace(-1.5, 0.5, num=5, base=dim)
            self.tick_times *= 2 * np.pi
        else:
            # Set x-ticks for unfolded data
            self.tick_times = np.logspace(-0.5, 1.5, num=5, base=dim)
            self.tick_times *= j_1_1 / E0

        # Set x-limits for unfolded data
        if self.xlim is None:
            self.xlim = (self.tick_times[0], self.tick_times[-1])

        # Set x-ticks for unfolded data
        if self.axes.xticks is None:
            self.axes.xticks = self.tick_times[1:-1]

    def plot(self):
        """Creates a plot of the CDO von Neumann entropy and saves it to a file."""
        # Create figure and axes
        self.create_figure()

        # Unpack entropy data
        times = self.data["times"]
        entropy = self.data["entropy"]

        # Create short notations for ensemble attributes
        dim = self.ensemble.dim

        # Normalize entropy w.r.t. maximum entropy
        entropy /= np.log(dim)

        # Set x- and y-scales to logarithmic
        self.ax.set_xscale("log", base=dim)

        # Turn off x- and y-axis minor ticks
        self.ax.xaxis.set_minor_locator(NullLocator())

        # Limit number of major ticks on x- and y-axis
        self.ax.xaxis.set_major_locator(LogLocator(base=dim, numticks=3))

        # Plot entropy
        self.ax.plot(
            times,
            entropy,
            color=self.e_color,
            alpha=self.e_alpha,
            linewidth=self.e_width,
            zorder=self.e_zorder,
            label=self.e_legend,
        )

        # Create vertical grid lines
        self.ax.vlines(
            self.tick_times[1:-1],
            self.ylim[0],
            self.ylim[1],
            color=self.grid_color,
            linestyle=self.grid_linestyle,
            linewidth=self.grid_linewidth,
            alpha=self.grid_alpha,
            zorder=self.grid_zorder,
        )

        # Finish plot and save it to a file
        self.set_plot(unfold=self.unfold)
