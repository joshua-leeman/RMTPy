# rmtpy/plotting/spectral_statistics_plots/spacing_histogram_plot/spacing_histogram_plot.py

# Postponed evaluation of annotations
from __future__ import annotations

# Standard library imports
from dataclasses import dataclass, field
from pathlib import Path

# Third-party imports
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Local application imports
from .spacing_histogram_axes import SpacingHistogramAxes
from .spacing_histogram_legend import SpacingHistogramLegend
from ..._base import Plot
from ....data.spectral_statistics_data import SpacingHistogramData
from ....ensembles import Ensemble, ManyBodyEnsemble, converter


# --------------------------------
# Spacing Histogram Plot Dataclass
# --------------------------------
@dataclass(repr=False, eq=False, kw_only=True)
class SpacingHistogramPlot(Plot):

    # Data to plot
    data: SpacingHistogramData

    # Unfolded data flag
    unfold: bool = False

    # Number of spacing histogram points
    num_points: int = 1000

    # Graphic colors
    hist_color: str = "Orange"
    spac_color: str = "Black"

    # Graphic alphas
    hist_alpha: float = 0.5
    spac_alpha: float = 1.0

    # Graphic widths
    spac_width: float = 2.0

    # Graphic z-orders
    hist_zorder: int = 1
    spac_zorder: int = 2

    # Legend labels
    hist_legend: str = "simulation"
    spac_legend: str = "surmise"

    # Legend handles
    handles: tuple[Patch, Line2D] = (
        Patch(color=hist_color, alpha=hist_alpha),
        Line2D([0], [0], color=spac_color, linewidth=spac_width),
    )

    # Legend labels
    labels: tuple[str, str] = (hist_legend, spac_legend)

    # Axes configuration
    axes: SpacingHistogramAxes = field(default_factory=SpacingHistogramAxes)

    # x-axis limits
    xlim: tuple[float, float] = (0.0, 4.0)  # factor of d

    # y-axis limits
    ylim: tuple[float, float] = (0.0, 1.2)  # factor of 1/d

    def __post_init__(self) -> None:
        """Initialize plot derived attributes after object creation."""

        # Initialize base class
        super(SpacingHistogramPlot, self).__post_init__()

        # Legend configuration
        self.legend = SpacingHistogramLegend(handles=self.handles, labels=self.labels)

        # Set derived attributes
        self.set_derived_attributes()

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

        # Set default legend title
        if self.legend.title is None:
            self.legend.title = self.ensemble.to_latex

        # Perform unfolding adjustments
        if self.unfold:
            # Prepend "unf_" to file name
            self.file_name = "unf_" + self.file_name

            # Append legend title with "unfolded"
            self.legend.title += "\nunfolded"

            # Replace axes labels with unfolded versions
            self.axes.xlabel = self.axes.unf_xlabel
            self.axes.ylabel = self.axes.unf_ylabel
            self.axes.xtick_labels = self.axes.unf_xtick_labels
            self.axes.ytick_labels = self.axes.unf_ytick_labels

            # Estimate global mean level spacing
            self.mean_spacing = 1.0
        else:
            # Estimate global mean level spacing
            self.mean_spacing = 2 * self.ensemble.E0 / self.ensemble.dim

        # Scale x-axis limits
        self.xlim = tuple(x * self.mean_spacing for x in self.xlim)

        # Scale x-axis ticks
        self.axes.xticks = tuple(x * self.mean_spacing for x in self.axes.xticks)
        self.axes.xticks_minor = tuple(
            x * self.mean_spacing for x in self.axes.xticks_minor
        )

        # Scale y-axis limits
        self.ylim = tuple(y / self.mean_spacing for y in self.ylim)

        # Scale y-axis ticks
        self.axes.yticks = tuple(y / self.mean_spacing for y in self.axes.yticks)
        self.axes.yticks_minor = tuple(
            y / self.mean_spacing for y in self.axes.yticks_minor
        )

    def plot(self) -> None:
        """Creates a nn-level spacings histogram and saves it to a file."""

        # Create figure and axes
        self.create_figure()

        # Unpack histogram data based on unfolding
        if self.unfold:
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

        # Create array of spacings for Wigner surmise
        spacings = np.linspace(0, self.xlim[1], self.num_points)

        # Calculate Wigner surmise
        surmise = self.ensemble.wigner_surmise(spacings / self.mean_spacing)
        surmise /= self.mean_spacing  # Scale by mean spacing

        # Plot Wigner surmise
        self.ax.plot(
            spacings,
            surmise,
            color=self.spac_color,
            linewidth=self.spac_width,
            alpha=self.spac_alpha,
            zorder=self.spac_zorder,
        )

        # Finalize and save plot
        self.finish_plot()
