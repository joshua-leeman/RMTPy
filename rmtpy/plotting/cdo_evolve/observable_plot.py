# rmtpy.plotting.cdo_evolution.observable_plot.py


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
class ObservablePlotAxes(PlotAxes):
    # x-axis labels
    xlabel: str = r"$Jt / j_{1,1}$"
    unf_xlabel: str = r"$\tau$"

    # y-axis labels
    ylabel: str = r"$0$"
    unf_ylabel: str = r"$1$"

    # x-tick labels
    xticklabels: tuple[str, ...] = (r"$1/N$", r"$\sqrt{D}/N$", r"$D/N$")
    unf_xticklabels: tuple[str, ...] = (r"$2\pi / D$", r"$2\pi / \sqrt{D}$", r"$2\pi$")

    # y-tick labels
    yticklabels: tuple[str, ...] = ()


# =======================================
# 3. Legend Dataclass
# =======================================
@dataclass(repr=False, eq=False, kw_only=True)
class ObservableLegend(PlotLegend):
    # Legend location
    loc: str = "upper right"
    bbox: tuple[float, float] = (0.75, 0.95)


# =======================================
# 4. Observable Plot Dataclass
# =======================================
@dataclass(repr=False, eq=False, kw_only=True)
class ObservablePlot(Plot):
    # Observable q-parameter
    obs_q: int = 2

    # Plot file name
    file_name: str = "observable_plot"

    # Expectation value curve parameters
    p0_color: str = "Blue"
    p0_alpha: float = 1.0
    p0_width: float = 1.0
    p0_zorder: int = 2
    p0_legend: str = r"$0$"

    # Error curve parameters
    p1_color: str = "Red"
    p1_alpha: float = 1.0
    p1_width: float = 1.0
    p1_zorder: int = 2
    p1_legend: str = r"$\sigma_A$"

    # Grid line parameters
    grid_color: str = rcParams["grid.color"]
    grid_linestyle: str = "dotted"
    grid_linewidth: float = rcParams["grid.linewidth"]
    grid_alpha: float = 1.0
    grid_zorder: int = 0

    # Legend handles and labels
    handles: tuple[Patch, Line2D] = (
        Line2D([0], [0], color=p0_color, alpha=p0_alpha, linewidth=p0_width),
        Line2D([0], [0], color=p1_color, alpha=p1_alpha, linewidth=p1_width),
    )
    labels: tuple[str, str] = (p0_legend, p1_legend)

    # Unfolded legend handles and labels
    unf_handles: tuple[Patch, Line2D, Line2D] = (
        Line2D([0], [0], color=p0_color, alpha=p0_alpha, linewidth=p0_width),
        Line2D([0], [0], color=p1_color, alpha=p1_alpha, linewidth=p1_width),
    )
    unf_labels: tuple[str, str, str] = (p0_legend, p1_legend)

    # Axes configuration
    axes: ObservablePlotAxes = field(default_factory=ObservablePlotAxes)

    def __post_init__(self):
        # Initialize base class
        super(ObservablePlot, self).__post_init__()

        # Set unfold flag from data
        object.__setattr__(self, "unfold", self.data["unfold"])

        # Switch legend handles and labels for unfolded data
        if self.unfold:
            self.handles = self.unf_handles
            self.labels = self.unf_labels

        # Set legend properties
        self.legend = ObservableLegend(handles=self.handles, labels=self.labels)

        # Set derived attributes
        self.set_derived_attributes()

    def set_derived_attributes(self) -> None:
        """Sets derived attributes for the plot."""
        # Create short notations for ensemble attributes
        N = self.ensemble.N
        dim = self.ensemble.dim
        E0 = self.ensemble.E0

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
            # Store first positive zero of 1st Bessel function
            j_1_1 = jn_zeros(1, 1)[0]

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
        """Creates a plot of expectation value and errors and saves it to a file."""
        # Create figure and axes
        self.create_figure()

        # Unpack data
        times = self.data["times"]
        obs_eigvals = self.data["obs_eigvals"]
        obs_expect = self.data["obs_expect"]
        obs_var = self.data["obs_var"]

        # Store maximum eigenvalues
        a_max = np.max(obs_eigvals)

        # Set y-limits
        if self.ylim is None:
            self.ylim = (-1.5, 1.5)

        # Set y-ticks
        if self.axes.yticks is None:
            self.axes.yticks = (-1.0, 0.0, 1.0)

        # Calculate observable standard deviation and normalize it
        obs_std = np.sqrt(obs_var)

        # Nomalize expectation value and standard deviation
        obs_expect /= a_max
        obs_std /= a_max

        # Create short notations for ensemble attributes
        dim = self.ensemble.dim

        # Calculate theoretical observable expectation and variance
        theo_expect = np.sum(obs_eigvals) / a_max / dim
        theo_std = np.sqrt(np.sum(obs_eigvals**2) / a_max**2 / dim - theo_expect**2)

        # Set x- and y-scales to logarithmic
        self.ax.set_xscale("log", base=dim)

        # Turn off x- and y-axis minor ticks
        self.ax.xaxis.set_minor_locator(NullLocator())

        # Limit number of major ticks on x- and y-axis
        self.ax.xaxis.set_major_locator(LogLocator(base=dim, numticks=3))

        # Calculate Chebyshev interval bounds
        upper_bounds = np.minimum(1, obs_expect + obs_std)
        lower_bounds = np.maximum(-1, obs_expect - obs_std)

        # Plot expectation value
        self.ax.plot(
            times,
            obs_expect,
            color=self.p0_color,
            alpha=self.p0_alpha,
            linewidth=self.p0_width,
            zorder=self.p0_zorder,
            label=self.p0_legend,
        )

        # Plot upper error
        self.ax.plot(
            times,
            upper_bounds.T,
            color=self.p1_color,
            alpha=self.p1_alpha,
            linewidth=self.p1_width,
            zorder=self.p1_zorder,
            label=self.p1_legend,
        )

        # Plot lower error
        self.ax.plot(
            times,
            lower_bounds.T,
            color=self.p1_color,
            alpha=self.p1_alpha,
            linewidth=self.p1_width,
            zorder=self.p1_zorder,
            label=self.p1_legend,
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

        # Create horizontal grid lines
        self.ax.hlines(
            [theo_expect, theo_expect + theo_std, theo_expect - theo_std],
            self.xlim[0],
            self.xlim[1],
            color=self.grid_color,
            linestyle=self.grid_linestyle,
            linewidth=self.grid_linewidth,
            alpha=self.grid_alpha,
            zorder=self.grid_zorder,
        )

        # Finish plot and save it to a file
        self.set_plot(unfold=self.unfold)
