# rmtpy/plotting/spectral_statistics_plots/spacing_histogram_plot/spacing_histogram_figure.py

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
from ....dataclasses.spectral_statistics_data import SpacingHistogramData
from ....ensembles import Ensemble, ManyBodyEnsemble, converter


# --------------------------------
# Spacing Histogram Plot Dataclass
# --------------------------------
@dataclass(repr=False, eq=False, kw_only=True)
class SpacingHistogramPlot(Plot):

    # Data to plot
    data: SpacingHistogramData

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

    def set_derived_attributes(self) -> None:
        """Set attributes that depend on simulation metadata."""

        # Configure legend
        self.legend = SpacingHistogramLegend(handles=self.handles, labels=self.labels)

        # Alias legend object
        legend = self.legend

        # Alias axes object
        axes = self.axes

        # Try to extract ensemble metadata
        try:
            ens_meta = self.data.metadata["simulation"]["args"]["ensemble"]
        except KeyError:
            raise ValueError("Ensemble metadata not found.")
        except TypeError:
            raise ValueError("Metadata is not properly structured.")

        # Initialize ensemble
        self.ensemble: ManyBodyEnsemble = converter.structure(ens_meta, Ensemble)

        # Alias ensemble and ensemble attributes
        ensemble = self.ensemble
        dim = ensemble.dim
        E0 = ensemble.E0

        # Insert universal class into legend if it exists
        if ensemble.univ_class is not None:
            self.spac_legend = f"{ensemble.univ_class} surmise"
            self.labels = (self.hist_legend, self.spac_legend)

        # Set default legend title
        if legend.title is None:
            legend.title = ensemble.to_latex

        # =================================================

        # Perform unfolding adjustments
        if self.unfold:
            # Prepend "unf_" to file name
            self.file_name = "unf_" + self.file_name

            # Append legend title with "unfolded"
            legend.title += "\nunfolded"

            # Replace axes labels with unfolded versions
            axes.xlabel = axes.unf_xlabel
            axes.ylabel = axes.unf_ylabel
            axes.xtick_labels = axes.unf_xtick_labels
            axes.ytick_labels = axes.unf_ytick_labels

            # Estimate global mean level spacing
            self.mean_spacing = 1.0
        else:
            # Estimate global mean level spacing
            self.mean_spacing = 2 * ensemble.E0 / ensemble.dim

        # Scale x-axis limits
        self.xlim = tuple(x * self.mean_spacing for x in self.xlim)

        # Scale x-axis ticks
        axes.xticks = tuple(x * self.mean_spacing for x in axes.xticks)
        axes.xticks_minor = tuple(x * self.mean_spacing for x in axes.xticks_minor)

        # Scale y-axis limits
        self.ylim = tuple(y / self.mean_spacing for y in self.ylim)

        # Scale y-axis ticks
        axes.yticks = tuple(y / self.mean_spacing for y in axes.yticks)
        axes.yticks_minor = tuple(y / self.mean_spacing for y in axes.yticks_minor)

    def plot(self, path: str | Path) -> None:
        """Generates and saves NN-level spacings histogram to a file."""

        # Set derived attributes
        self.set_derived_attributes()

        # Unpack and alias histogram data based on unfolding
        if self.unfold:
            bin_edges = self.data.unf_bins
            counts = self.data.unf_hist
        else:
            bin_edges = self.data.bins
            counts = self.data.hist

        # Alias ensemble
        ensemble = self.ensemble

        # =================================================

        # Create figure and axes
        self.create_figure()

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
        surmise = ensemble.wigner_surmise(spacings / self.mean_spacing)
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
        self.finish_plot(path=path)
