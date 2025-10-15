# rmtpy/plotting/spectral_statistics_plots/form_factors_plot/form_factors_figure.py

# Postponed evaluation of annotations
from __future__ import annotations

# Standard library imports
from dataclasses import dataclass, field
from pathlib import Path

# Third-party imports
import numpy as np
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullLocator
from scipy.special import jn_zeros

# Local application imports
from .form_factors_axes import FormFactorsAxes
from .form_factors_legend import FormFactorsLegend
from ..._base import Plot
from ....data.spectral_statistics_data import FormFactorsData
from ....ensembles import Ensemble, ManyBodyEnsemble, converter


# ------------------------------------
# Spectral Form Factors Plot Dataclass
# ------------------------------------
@dataclass(repr=False, eq=False, kw_only=True)
class FormFactorsPlot(Plot):

    # Data to plot
    data: FormFactorsData

    # Number of universal form factor points
    num_points: int = 1000

    # Graphic colors
    grid_color: str = rcParams["grid.color"]
    sff_color: str = "Blue"
    csff_color: str = "Red"
    usff_color: str = "Black"

    # Graphic alphas
    grid_alpha: float = 1.0
    sff_alpha: float = 1.0
    csff_alpha: float = 1.0
    usff_alpha: float = 1.0

    # Graphic widths
    grid_width: float = rcParams["grid.linewidth"]
    sff_width: float = 0.5
    csff_width: float = 0.5
    usff_width: float = 0.5

    # Graphic styles
    grid_linestyle: str = "dotted"

    # Graphic z-orders
    grid_zorder: int = 0
    sff_zorder: int = 2
    csff_zorder: int = 2
    usff_zorder: int = 2

    # Legend labels
    sff_legend: str = "SFF"
    csff_legend: str = "cSFF"
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
    axes: FormFactorsAxes = field(default_factory=FormFactorsAxes)

    # x-axis limits
    xlim: tuple[float, float] = (-0.5, 1.5)  # base dim log scale (factor of j_1_1 / E0)
    unf_xlim: tuple[float, float] = (-1.5, 0.5)  # base unf log scale (factor of 2 pi)

    # y-axis limits
    ylim: tuple[float, float] = (-2.2, 0.2)  # base dim log scale

    def set_derived_attributes(self) -> None:
        """Set attributes that depend on simulation metadata."""

        # Try to extract ensemble metadata
        try:
            ens_meta = self.data.metadata["simulation"]["args"]["ensemble"]
        except KeyError:
            raise ValueError("Ensemble metadata not found.")
        except TypeError:
            raise ValueError("Metadata is not properly structured.")

        # Initialize ensemble
        self.ensemble: ManyBodyEnsemble = converter.structure(ens_meta, Ensemble)

        # Alias ensemble attributes
        dim = self.ensemble.dim
        E0 = self.ensemble.E0

        # Store first positive zero of 1st Bessel function
        j_1_1 = jn_zeros(1, 1)[0]

        # Set default legend title
        if self.legend.title is None:
            self.legend.title = self.ensemble.to_latex

        # Set y-limits
        self.ylim = tuple(dim**y for y in self.ylim)

        # Set y-axis ticks
        self.axes.yticks = tuple(dim**y for y in self.axes.yticks)

        # Perform unfolding adjustments
        if self.unfold:
            # Prepend "unf_" to file name
            self.file_name = "unf_" + self.file_name

            # Append legend title with "unfolded"
            self.legend.title += "\nunfolded"

            # Insert universal class into usff_label if it exists
            if self.ensemble.univ_class is not None:
                self.usff_legend = f"{self.ensemble.univ_class} limit"
                self.unf_labels = (self.sff_legend, self.csff_legend, self.usff_legend)

            # Configure legend for unfolded data
            self.legend = FormFactorsLegend(
                handles=self.unf_handles, labels=self.unf_labels
            )

            # Change legend location
            self.legend.bbox = self.legend.unf_bbox

            # Replace axes labels with unfolded versions
            self.axes.xlabel = self.axes.unf_xlabel
            self.axes.ylabel = self.axes.unf_ylabel
            self.axes.xtick_labels = self.axes.unf_xtick_labels

            # Replace and scale x-limits
            self.xlim = tuple(dim**x * 2 * np.pi for x in self.unf_xlim)

            # Replace and scale x-ticks
            self.axes.xticks = tuple(dim**x * 2 * np.pi for x in self.axes.unf_xticks)
        else:
            # Configure legend data
            self.legend = FormFactorsLegend(handles=self.handles, labels=self.labels)

            # Scale x-limits
            self.xlim = tuple(dim**x * j_1_1 / E0 for x in self.xlim)

            # Scale x-ticks
            self.axes.xticks = tuple(dim**x * j_1_1 / E0 for x in self.axes.xticks)

    def plot(self, path: str | Path) -> None:
        """Generates and saves form factors plot to file."""

        # Set derived attributes
        self.set_derived_attributes()

        # Create figure and axes
        self.create_figure()

        # Unpack form factor data based on unfolding
        if self.unfold:
            # Unfolded data
            times = self.data.unf_times
            sff = self.data.unf_sff
            csff = self.data.unf_csff
        else:
            # Regular data
            times = self.data.times
            sff = self.data.sff
            csff = self.data.csff

        # Alias ensemble dimension
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

            # Plot universal connected spectral form factor
            self.ax.plot(
                times,
                usff,
                color=self.usff_color,
                alpha=self.usff_alpha,
                linewidth=self.usff_width,
                zorder=self.usff_zorder,
                label=self.usff_legend,
            )

        # Add grid lines on major x-ticks
        self.ax.vlines(
            self.axes.xticks,
            ymin=self.ylim[0],
            ymax=self.ylim[1],
            colors=self.grid_color,
            linestyles=self.grid_linestyle,
            linewidth=self.grid_width,
            alpha=self.grid_alpha,
            zorder=self.grid_zorder,
        )

        # Finish plot and save it to a file
        self.finish_plot(path)
