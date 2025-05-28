# rmtpy.plotting.spectral_statistics.spectral_histogram.py


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
class SpectralHistogramAxes(PlotAxes):
    # Axes labels
    xlabel: str = r"$E$"
    ylabel: str = r"$\ensavg{f(E)}$"
    unf_xlabel: str = r"$\xi$"
    unf_ylabel: str = r"$\ensavg{f(\xi)}$"

    # Axes tick labels
    xticklabels: tuple[str, ...] = (r"$-NJ$", r"$0$", r"$NJ$")
    yticklabels: tuple[str, ...] = (
        r"$0$",
        r"$\frac{1}{\pi NJ}$",
        r"$\frac{2}{\pi NJ}$",
    )
    unf_xticklabels: tuple[str, ...] = (r"$-\frac{1}{2}D$", r"$0$", r"$\frac{1}{2}D$")
    unf_yticklabels: tuple[str, ...] = (r"$0$", r"$\frac{1}{2}D^{-1}$", r"$D^{-1}$")

    # Define Poisson specific tick labels
    poi_yticklabels: tuple[str, ...] = (r"$0$", r"$\frac{1}{4NJ}$", r"$\frac{1}{2NJ}$")

    # Define SYK q=2 and q=4 specific tick labels
    syk2_yticklabels: tuple[str, ...] = (
        r"$0$",
        r"$\frac{1}{\pi NJ}$",
        r"$\frac{2}{\pi NJ}$",
        r"$\frac{3}{\pi NJ}$",
        r"$\frac{4}{\pi NJ}$",
        r"$\frac{5}{\pi NJ}$",
    )
    syk4_yticklabels: tuple[str, ...] = (
        r"$0$",
        r"$\frac{1}{\pi NJ}$",
        r"$\frac{2}{\pi NJ}$",
        r"$\frac{3}{\pi NJ}$",
    )


# =======================================
# 3. Legend Dataclass
# =======================================
@dataclass(repr=False, eq=False, kw_only=True)
class SpectralHistogramLegend(PlotLegend):
    # Legend location
    loc: str = "upper right"
    bbox: tuple[float, float] = (0.94, 0.95)


# =======================================
# 4. Spectral Histogram Dataclass
# =======================================
@dataclass(repr=False, eq=False, kw_only=True)
class SpectralHistogram(Plot):
    # File name
    file_name: str = "spectral_histogram.png"

    # Unfolded data flag
    unfold: bool = False

    # Number of spectral density points
    n_points: int = 1000

    # Histogram graphic parameters
    hist_color: str = "RoyalBlue"
    hist_alpha: float = 0.5
    hist_zorder: int = 1
    hist_legend: str = "simulation"

    # Spectral density graphic parameters
    spec_color: str = "Black"
    spec_alpha: float = 1.0
    spec_width: float = 2.0
    spec_zorder: int = 2
    spec_legend: str = "theory"

    # Legend handles and labels
    handles: tuple[Patch, Line2D] = (
        Patch(color=hist_color, alpha=hist_alpha),
        Line2D([0], [0], color=spec_color, linewidth=spec_width),
    )
    labels: tuple[str, str] = (hist_legend, spec_legend)

    # Axes configuration
    axes: SpectralHistogramAxes = field(default_factory=SpectralHistogramAxes)

    def __post_init__(self):
        # Initialize base class
        super(SpectralHistogram, self).__post_init__()

        # Set legend properties
        self.legend = SpectralHistogramLegend(handles=self.handles, labels=self.labels)

        # Set derived attributes
        self.set_derived_attributes()

    def set_derived_attributes(self) -> None:
        """Sets derived attributes for the plot."""
        # Create short notations for ensemble attributes
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
            self.axes.yticklabels = self.axes.unf_yticklabels

            # Append legend title with "unfolded"
            self.legend.title += "\nunfolded"

            # Set limits for unfolded data
            if self.xlim is None:
                self.xlim = (-1.6 * dim / 2, 1.6 * dim / 2)
            if self.ylim is None:
                self.ylim = (0, 1.5 / dim)

            # Set ticks for unfolded data
            if self.axes.xticks is None:
                self.axes.xticks = (-dim / 2, 0, dim / 2)
            if self.axes.xticks_minor is None:
                self.axes.xticks_minor = np.arange(-1.5, 2.5, 1) * dim / 2
            if self.axes.yticks is None:
                self.axes.yticks = (0, 0.5 / dim, 1 / dim)
            if self.axes.yticks_minor is None:
                self.axes.yticks_minor = np.arange(0.25, 1.75, 0.5) / dim
        else:
            # Set limits for non-unfolded data
            if self.xlim is None:
                self.xlim = (-1.2 * E0, 1.2 * E0)
            if self.ylim is None:
                self.ylim = (0, 2.6 / np.pi / E0)

            # Set ticks for non-unfolded data
            if self.axes.xticks is None:
                self.axes.xticks = (-E0, 0, E0)
            if self.axes.xticks_minor is None:
                self.axes.xticks_minor = (-E0 / 2, E0 / 2)
            if self.axes.yticks is None:
                self.axes.yticks = np.arange(0, 3, 1) / np.pi / E0
            if self.axes.yticks_minor is None:
                self.axes.yticks_minor = np.arange(0.5, 3.5, 1) / np.pi / E0

            # Rewrites attributes for Poisson special case
            if self.ensemble.__class__.__name__ == "Poisson":
                self.ylim = (0, 0.75 / E0)
                self.axes.yticks = (0, 0.25 / E0, 0.5 / E0)
                self.axes.yticks_minor = (0.125 / E0, 0.375 / E0, 0.625 / E0)
                self.axes.yticklabels = self.axes.poi_yticklabels

            # Rewrites attributes for Poisson special case
            if self.ensemble.__class__.__name__ == "SYK":
                if self.ensemble.q == 2:
                    self.ylim = (0, 5 / np.pi / E0)
                    self.axes.yticks = np.arange(0, 6, 1) / np.pi / E0
                    self.axes.yticks_minor = np.arange(0.5, 6, 0.5) / np.pi / E0
                    self.axes.yticklabels = self.axes.syk2_yticklabels
                elif self.ensemble.q == 4:
                    self.ylim = (0, 3 / np.pi / E0)
                    self.axes.yticks = np.arange(0, 4, 1) / np.pi / E0
                    self.axes.yticks_minor = np.arange(0.5, 4, 0.5) / np.pi / E0
                    self.axes.yticklabels = self.axes.syk4_yticklabels

    def plot(self):
        """Creates a spectral histogram and saves it to a file."""
        # Create figure and axes
        self.create_figure()

        # Unpack histogram data based on unfolding
        if self.unfold:
            # Unfolded histogram data
            bin_edges = self.data["unf_spec_bin_edges"]
            counts = self.data["unf_spec_hist"]
        else:
            bin_edges = self.data["spec_bin_edges"]
            counts = self.data["spec_hist"]

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

        # Create array of energy values for spectral density
        energies = np.linspace(self.xlim[0], self.xlim[1], self.n_points)

        # Calculate spectral density based on unfolding
        if self.unfold:
            # Unfolded spectral density
            spec_pdf = np.zeros(self.n_points)
            spec_pdf[np.abs(energies) < dim / 2] = 1 / dim
        else:
            # Spectral density
            spec_pdf = self.ensemble.pdf(energies)

        # Plot spectral density
        self.ax.plot(
            energies,
            spec_pdf,
            color=self.spec_color,
            alpha=self.spec_alpha,
            linewidth=self.spec_width,
            zorder=self.spec_zorder,
        )

        # Finishes plot and saves it to a file
        self.set_plot(unfold=self.unfold)
