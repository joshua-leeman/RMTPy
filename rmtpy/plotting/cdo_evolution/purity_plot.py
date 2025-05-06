# rmtpy.plotting.cdo_evolution.purity_plot.py


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
class PurityPlotAxes(PlotAxes):
    # x-axis labels
    xlabel: str = r"$Jt / j_{1,1}$"
    unf_xlabel: str = r"$\tau$"

    # y-axis labels
    ylabel: str = r"$\gamma(t)$"
    unf_ylabel: str = r"$\gamma(\tau)$"

    # x-tick labels
    xticklabels: tuple[str, ...] = (r"$1/N$", r"$\sqrt{D}/N$", r"$D/N$")
    unf_xticklabels: tuple[str, ...] = (r"$2\pi / D$", r"$2\pi / \sqrt{D}$", r"$2\pi$")

    # y-tick labels
    yticklabels: tuple[str, ...] = (r"$D^{-2}$", r"$D^{-1}$", r"$1$")


# =======================================
# 3. Legend Dataclass
# =======================================
@dataclass(repr=False, eq=False, kw_only=True)
class PurityLegend(PlotLegend):
    # Legend location
    loc: str = "upper right"
    bbox: tuple[float, float] = (0.75, 0.95)


# =======================================
# 4. Purity Plot Dataclass
# =======================================
@dataclass(repr=False, eq=False, kw_only=True)
class PurityPlot(Plot):
    # Unfolded data flag
    unfold: bool = False

    # Plot file name
    plot_filename: str = "purity_plot"

    # Quantum purity curve parameters
    qp_color: str = "Blue"
    qp_alpha: float = 1.0
    qp_width: float = 1.0
    qp_zorder: int = 2
    qp_legend: str = "quantum"

    # Classical purity curve parameters
    cp_color: str = "Red"
    cp_alpha: float = 1.0
    cp_width: float = 1.0
    cp_zorder: int = 2
    cp_legend: str = "classical"

    # Grid line parameters
    grid_color: str = rcParams["grid.color"]
    grid_linestyle: str = "dotted"
    grid_linewidth: float = rcParams["grid.linewidth"]
    grid_alpha: float = 1.0
    grid_zorder: int = 0

    # Legend handles and labels
    handles: tuple[Patch, Line2D] = (
        Line2D([0], [0], color=qp_color, alpha=qp_alpha, linewidth=qp_width),
        Line2D([0], [0], color=cp_color, alpha=cp_alpha, linewidth=cp_width),
    )
    labels: tuple[str, str] = (qp_legend, cp_legend)

    # Unfolded legend handles and labels
    unf_handles: tuple[Patch, Line2D, Line2D] = (
        Line2D([0], [0], color=qp_color, alpha=qp_alpha, linewidth=qp_width),
        Line2D([0], [0], color=cp_color, alpha=cp_alpha, linewidth=cp_width),
    )
    unf_labels: tuple[str, str, str] = (qp_legend, cp_legend)

    # Axes configuration
    axes: PurityPlotAxes = field(default_factory=PurityPlotAxes)

    def __post_init__(self):
        # Initialize base class
        super(PurityPlot, self).__post_init__()

        # Switch legend handles and labels for unfolded data
        if self.unfold:
            self.handles = self.unf_handles
            self.labels = self.unf_labels

        # Set legend properties
        self.legend = PurityLegend(handles=self.handles, labels=self.labels)

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
            self.ylim = (dim**-2.2, dim**0.2)

        # Set y-ticks
        if self.axes.yticks is None:
            self.axes.yticks = (dim**-2, dim**-1, 1)

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
        """Creates plots of quantum and classical purities and saves it to a file."""
        # Create figure and axes
        self.create_figure()

        # Unpack purity data
        times = self.data["times"]
        q_purity = self.data["quantum_purity"]
        c_purity = self.data["classical_purity"]

        # Create short notations for dimension
        dim = self.ensemble.dim

        # Set x- and y-scales to logarithmic
        self.ax.set_xscale("log", base=dim)
        self.ax.set_yscale("log", base=dim)

        # Turn off x- and y-axis minor ticks
        self.ax.xaxis.set_minor_locator(NullLocator())
        self.ax.yaxis.set_minor_locator(NullLocator())

        # Limit number of major ticks on x- and y-axis
        self.ax.xaxis.set_major_locator(LogLocator(base=dim, numticks=3))
        self.ax.yaxis.set_major_locator(LogLocator(base=dim, numticks=3))

        # Plot quantum purity
        self.ax.plot(
            times,
            q_purity,
            color=self.qp_color,
            alpha=self.qp_alpha,
            linewidth=self.qp_width,
            zorder=self.qp_zorder,
            label=self.qp_legend,
        )

        # Plot classical purity
        self.ax.plot(
            times,
            c_purity,
            color=self.cp_color,
            alpha=self.cp_alpha,
            linewidth=self.cp_width,
            zorder=self.cp_zorder,
            label=self.cp_legend,
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
        self.set_plot()
