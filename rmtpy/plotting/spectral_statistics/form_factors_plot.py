# rmtpy.plotting.spectral_statistics.form_factors_plot.py


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
class FormFactorPlotAxes(PlotAxes):
    # x-axis labels
    xlabel: str = r"$Jt / j_{1,1}$"
    unf_xlabel: str = r"$\tau$"

    # y-axis labels
    ylabel: str = r"$K(t)$"
    unf_ylabel: str = r"$K(\tau)$"

    # x-tick labels
    xticklabels: tuple[str, ...] = (r"$1/N$", r"$\sqrt{D}/N$", r"$D/N$")
    unf_xticklabels: tuple[str, ...] = (r"$2\pi / D$", r"$2\pi / \sqrt{D}$", r"$2\pi$")

    # y-tick labels
    yticklabels: tuple[str, ...] = (r"$D^{-2}$", r"$D^{-1}$", r"$1$")


# =======================================
# 3. Legend Dataclass
# =======================================
@dataclass(repr=False, eq=False, kw_only=True)
class FormFactorPlotLegend(PlotLegend):
    # Legend location
    loc: str = "upper right"
    bbox: tuple[float, float] = (0.735, 0.95)


# =======================================
# 4. Form Factor Plot Dataclass
# =======================================
@dataclass(repr=False, eq=False, kw_only=True)
class FormFactorPlot(Plot):
    # File name
    file_name: str = "form_factors.png"

    # Unfolded data flag
    unfold: bool = False

    # Number of universal curve points
    n_points: int = 1000

    # SFF curve parameters
    sff_color: str = "Blue"
    sff_alpha: float = 1.0
    sff_width: float = 1.0
    sff_zorder: int = 2
    sff_legend: str = "SFF"

    # cSFF curve parameters
    csff_color: str = "Red"
    csff_alpha: float = 1.0
    csff_width: float = 1.0
    csff_zorder: int = 2
    csff_legend: str = "cSFF"

    # Grid line parameters
    grid_color: str = rcParams["grid.color"]
    grid_linestyle: str = "dotted"
    grid_linewidth: float = rcParams["grid.linewidth"]
    grid_alpha: float = 1.0
    grid_zorder: int = 0

    # Universal curve parameters
    usff_color: str = "Black"
    usff_alpha: float = 1.0
    usff_width: float = 1.0
    usff_zorder: int = 2
    usff_legend: str = "universal"

    # Legend handles and labels
    handles: tuple[Patch, Line2D] = (
        Line2D([0], [0], color=sff_color, alpha=sff_alpha, linewidth=sff_width),
        Line2D([0], [0], color=csff_color, alpha=csff_alpha, linewidth=csff_width),
    )
    labels: tuple[str, str] = (sff_legend, csff_legend)

    # Unfolded legend handles and labels
    unf_handles: tuple[Patch, Line2D, Line2D] = (
        Line2D([0], [0], color=sff_color, alpha=sff_alpha, linewidth=sff_width),
        Line2D([0], [0], color=csff_color, alpha=csff_alpha, linewidth=csff_width),
        Line2D([0], [0], color=usff_color, alpha=usff_alpha, linewidth=usff_width),
    )
    unf_labels: tuple[str, str, str] = (sff_legend, csff_legend, usff_legend)

    # Axes configuration
    axes: FormFactorPlotAxes = field(default_factory=FormFactorPlotAxes)

    def __post_init__(self):
        # Initialize base class
        super(FormFactorPlot, self).__post_init__()

        # Switch legend handles and labels for unfolded data
        if self.unfold:
            # Insert universal class into usff_label if it exists
            if self.ensemble.univ_class is not None:
                self.usff_legend = f"{self.ensemble.univ_class} limit"
                self.unf_labels = (self.sff_legend, self.csff_legend, self.usff_legend)

            # Set legend handles and labels for unfolded data
            self.handles = self.unf_handles
            self.labels = self.unf_labels

        # Set legend properties
        self.legend = FormFactorPlotLegend(handles=self.handles, labels=self.labels)

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
        """Creates a plot of the spectral form factors and saves it to a file."""
        # Create figure and axes
        self.create_figure()

        # Unpack form factor data based on unfolding
        if self.unfold:
            # Unfolded histogram data
            times = self.data["unf_times"]
            sff = self.data["unf_sff"]
            csff = self.data["unf_csff"]
        else:
            # Non-unfolded histogram data
            times = self.data["times"]
            sff = self.data["sff"]
            csff = self.data["csff"]

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

        # Plot spectral form factor
        self.ax.plot(
            times,
            sff,
            color=self.sff_color,
            alpha=self.sff_alpha,
            linewidth=self.sff_width,
            zorder=self.sff_zorder,
            label=self.sff_legend,
        )

        # Plot connected spectral form factor
        self.ax.plot(
            times,
            csff,
            color=self.csff_color,
            alpha=self.csff_alpha,
            linewidth=self.csff_width,
            zorder=self.csff_zorder,
            label=self.csff_legend,
        )

        # If unfolded, plot universal curve
        if self.unfold:
            # Generate universal curve
            usff = self.ensemble.univ_csff(times)

            # Plot universal curve
            self.ax.plot(
                times,
                usff,
                color=self.usff_color,
                alpha=self.usff_alpha,
                linewidth=self.usff_width,
                zorder=self.usff_zorder,
                label=self.usff_legend,
            )
        # If not unfolded, plot grid lines on major x-ticks
        else:
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
