# rmtpy.plotting.spectral_statistics.spacings_histogram.py

# =======================================
# 1. Imports
# =======================================
# Standard library imports
from __future__ import annotations
from dataclasses import dataclass, field

# Third-party imports
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Local application imports
from rmtpy.plotting._plot import Plot, PlotAxes, PlotLegend


# =======================================
# 2. Axes Dataclass
# =======================================
@dataclass(repr=False, eq=False, kw_only=True)
class SpacingsHistogramAxes(PlotAxes):
    # Axes labels
    xlabel: str = r"$\Delta E / d$"
    ylabel: str = r"$\ensavg{f(\Delta E)}$"
    unf_xlabel: str = r"$s$"
    unf_ylabel: str = r"$\ensavg{f(s)}$"

    # Axes tick labels
    xticklabels: tuple[str, ...] = (r"$0$", r"$d$", r"$2d$", r"$3d$", r"$4d$")
    yticklabels: tuple[str, ...] = (r"$0$", r"$\frac{1}{2}d^{-1}$", r"$d^{-1}$")
    unf_xticklabels: tuple[str, ...] = (r"$0$", r"$1$", r"$2$", r"$3$", r"$4$")
    unf_yticklabels: tuple[str, ...] = (r"$0$", r"$\frac{1}{2}$", r"$1$")


# =======================================
# 3. Legend Dataclass
# =======================================
@dataclass(repr=False, eq=False, kw_only=True)
class SpacingsHistogramLegend(PlotLegend):
    # Legend location
    loc: str = "upper right"
    bbox: tuple[float, float] = (0.92, 0.92)


# =======================================
# 4. Spacings Histogram Dataclass
# =======================================
@dataclass(repr=False, eq=False, kw_only=True)
class SpacingsHistogram(Plot):
    # File name
    file_name: str = "spacings_histogram.png"

    # Unfolded data flag
    unfold: bool = False

    # Number of surmise points
    n_points: int = 1000

    # Histogram graphic parameters
    hist_color: str = "Orange"
    hist_alpha: float = 0.5
    hist_zorder: int = 1
    hist_legend: str = "simulation"

    # Wigner surmise graphic parameters
    spac_color: str = "Black"
    spac_alpha: float = 1.0
    spac_width: float = 2.0
    spac_zorder: int = 2
    spac_legend: str = "surmise"

    # Legend handles and labels
    handles: tuple[Patch, Line2D] = (
        Patch(color=hist_color, alpha=hist_alpha),
        Line2D([0], [0], color=spac_color, linewidth=spac_width),
    )
    labels: tuple[str, str] = (hist_legend, spac_legend)

    # Axes configuration
    axes: SpacingsHistogramAxes = field(default_factory=SpacingsHistogramAxes)

    def __post_init__(self):
        # Initialize base class
        super(SpacingsHistogram, self).__post_init__()

        # Insert universal class into surmise label if it exists
        if self.ensemble.univ_class is not None:
            self.spac_legend = f"{self.ensemble.univ_class} surmise"
            self.labels = (self.hist_legend, self.spac_legend)

        # Set legend properties
        self.legend = SpacingsHistogramLegend(handles=self.handles, labels=self.labels)

        # Set derived attributes
        self.set_derived_attributes()

    def set_derived_attributes(self) -> None:
        """Sets derived attributes for the plot."""
        # Create short notations for ensemble attributes
        dim = self.ensemble.dim
        E0 = self.ensemble.E0

        # Estimate global mean level spacing
        if self.unfold:
            global_mean_spacing = 1.0
        else:
            global_mean_spacing = 2 * E0 / dim

        # Set x-limits
        if self.xlim is None:
            self.xlim = (0, 4 * global_mean_spacing)

        # Set y-limits
        if self.ylim is None:
            self.ylim = (0, 1.2 / global_mean_spacing)

        # Set x-ticks
        if self.axes.xticks is None:
            self.axes.xticks = np.array(range(5)) * global_mean_spacing
        if self.axes.xticks_minor is None:
            self.axes.xticks_minor = (np.array(range(4)) + 0.5) * global_mean_spacing

        # Set y-ticks
        if self.axes.yticks is None:
            self.axes.yticks = np.array([0, 0.5, 1]) / global_mean_spacing
        if self.axes.yticks_minor is None:
            self.axes.yticks_minor = np.array([0.25, 0.75]) / global_mean_spacing

        # Set default legend title
        if self.legend.title is None:
            pass
            self.legend.title = self.ensemble._to_latex()

        # Configure plot based on unfolding
        if self.unfold:
            # Replace attributes for unfolded data
            self.axes.xlabel = self.axes.unf_xlabel
            self.axes.ylabel = self.axes.unf_ylabel
            self.axes.xticklabels = self.axes.unf_xticklabels
            self.axes.yticklabels = self.axes.unf_yticklabels

            # Append legend title with "unfolded"
            self.legend.title += "\nunfolded"

    def plot(self):
        """Creates a nn-level spacings histogram and saves it to a file."""
        # Create figure and axes
        self.create_figure()

        # Unpack histogram data based on unfolding
        if self.unfold:
            # Unfolded histogram data
            bin_edges = self.data["unf_spac_bin_edges"]
            counts = self.data["unf_spac_hist"]
        else:
            bin_edges = self.data["spac_bin_edges"]
            counts = self.data["spac_hist"]

        # Plot histogram
        self.ax.hist(
            bin_edges[:-1],
            bins=bin_edges,
            weights=counts,
            color=self.hist_color,
            alpha=self.hist_alpha,
            zorder=self.hist_zorder,
        )

        # Create short notations for ensemble attributes
        dim = self.ensemble.dim
        E0 = self.ensemble.E0

        # Estimate global mean level spacing
        if self.unfold:
            global_mean_spacing = 1.0
        else:
            global_mean_spacing = 2 * E0 / dim

        # Create array of spacings values for Wigner surmise
        spacings = np.linspace(self.xlim[0], self.xlim[1], self.n_points)

        # Calculate Wigner surmise
        spac_pdf = self.ensemble.wigner_surmise(spacings / global_mean_spacing)
        spac_pdf /= global_mean_spacing

        # Plot spectral density
        self.ax.plot(
            spacings,
            spac_pdf,
            color=self.spac_color,
            alpha=self.spac_alpha,
            linewidth=self.spac_width,
            zorder=self.spac_zorder,
        )

        # Finishes plot and saves it to a file
        self.set_plot(unfold=self.unfold)
