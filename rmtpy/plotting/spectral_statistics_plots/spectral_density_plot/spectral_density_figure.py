# rmtpy/plotting/spectral_statistics_plots/spectral_density_plot/spectral_density_figure.py

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
from .spectral_density_axes import SpectralDensityAxes
from .spectral_density_legend import SpectralDensityLegend
from ..._base import Plot
from ....dataclasses.spectral_statistics_data import SpectralDensityData
from ....ensembles import Ensemble, ManyBodyEnsemble, converter
from ....ensembles.poisson import Poisson
from ....ensembles.syk import SYK


# -------------------------------
# Spectral Density Plot Dataclass
# -------------------------------
@dataclass(repr=False, eq=False, kw_only=True)
class SpectralDensityPlot(Plot):

    # Data to plot
    data: SpectralDensityData

    # Number of spectral density points
    num_points: int = 1000

    # Graphic colors
    hist_color: str = "RoyalBlue"
    spec_color: str = "Black"

    # Graphic alphas
    hist_alpha: float = 0.5
    spec_alpha: float = 1.0

    # Graphic widths
    spec_width: float = 2.0

    # Graphic z-orders
    hist_zorder: int = 1
    spec_zorder: int = 2

    # Legend labels
    hist_legend: str = "simulation"
    spec_legend: str = "theory"

    # Legend handles
    handles: tuple[Patch, Line2D] = (
        Patch(color=hist_color, alpha=hist_alpha),
        Line2D([0], [0], color=spec_color, linewidth=spec_width),
    )

    # Legend labels
    labels: tuple[str, str] = (hist_legend, spec_legend)

    # Axes configuration
    axes: SpectralDensityAxes = field(default_factory=SpectralDensityAxes)

    # x-axis limits
    xlim: tuple[float, float] = (-1.2, 1.2)  # factor of E0
    unf_xlim: tuple[float, float] = (-1.2, 1.2)  # factor of D/2

    # y-axis limits
    ylim: tuple[float, float] = (0.0, 2.6)  # factor of 1/pi/(E0)
    unf_ylim: tuple[float, float] = (0.0, 1.625)  # factor of 1/D

    # Poisson specific y-axis limits
    poi_ylim: tuple[float, float] = (0.0, 1.25)  # factor of 1/(E0)

    # SYK q=2 specific y-axis limits
    syk2_ylim: tuple[float, float] = (0.0, 4.0)  # factor of 1/pi/(E0)

    # SYK q=4 specific y-axis limits
    syk4_ylim: tuple[float, float] = (0.0, 2.5)  # factor of 1/pi/(E0)

    def set_derived_attributes(self) -> None:
        """Set attributes that depend on simulation metadata."""

        # Configure legend
        self.legend = SpectralDensityLegend(handles=self.handles, labels=self.labels)

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

        # Set default legend title
        if legend.title is None:
            legend.title = ensemble.to_latex

        # =================================================

        # Configure plot based on unfolding
        if self.unfold:
            # Prepend "unf_" to file name
            self.file_name = "unf_" + self.file_name

            # Scale unfolded x-axis limits
            self.unf_xlim = tuple(x * dim / 2 for x in self.unf_xlim)

            # Scale unfolded y-axis limits
            self.unf_ylim = tuple(y / dim for y in self.unf_ylim)

            # Scale unfolded x-axis ticks and minor ticks
            axes.unf_xticks = tuple(x * dim / 2 for x in axes.unf_xticks)
            axes.unf_xticks_minor = tuple(x * dim / 2 for x in axes.unf_xticks_minor)

            # Scale unfolded y-axis ticks and minor ticks
            axes.unf_yticks = tuple(y / dim for y in axes.unf_yticks)
            axes.unf_yticks_minor = tuple(y / dim for y in axes.unf_yticks_minor)

            # Append legend title with "unfolded"
            legend.title += "\nunfolded"

            # Replace x-axis limits for unfolded data
            self.xlim = self.unf_xlim
            self.ylim = self.unf_ylim

            # Replace axes labels for unfolded data
            axes.xlabel = axes.unf_xlabel
            axes.ylabel = axes.unf_ylabel
            axes.xtick_labels = axes.unf_xtick_labels
            axes.ytick_labels = axes.unf_ytick_labels

            # Replace axes ticks for unfolded data
            axes.xticks = axes.unf_xticks
            axes.xticks_minor = axes.unf_xticks_minor
            axes.yticks = axes.unf_yticks
            axes.yticks_minor = axes.unf_yticks_minor

        # Else, configure for regular data
        else:
            # Rewrite y-axis attributes for Poisson ensemble
            if type(self.ensemble) == Poisson:
                # Set y-axis limits for Poisson ensemble
                self.ylim = self.poi_ylim

                # Set y-axis ticks and labels for Poisson ensemble
                axes.ytick_labels = axes.poi_ytick_labels
                axes.yticks = axes.poi_yticks
                axes.yticks_minor = axes.poi_yticks_minor

            # Rewrite y-axis attributes for SYK ensembles
            elif type(self.ensemble) == SYK:
                if self.ensemble.q == 2:
                    # Set y-axis limits for SYK q=2 ensemble
                    self.ylim = self.syk2_ylim

                    # Set y-axis ticks and labels for SYK q=2 ensemble
                    axes.ytick_labels = axes.syk2_ytick_labels
                    axes.yticks = axes.syk2_yticks
                    axes.yticks_minor = axes.syk2_yticks_minor

                elif self.ensemble.q == 4:
                    # Set y-axis limits for SYK q=4 ensemble
                    self.ylim = self.syk4_ylim

                    # Set y-axis ticks and labels for SYK q=4 ensemble
                    axes.ytick_labels = axes.syk4_ytick_labels
                    axes.yticks = axes.syk4_yticks
                    axes.yticks_minor = axes.syk4_yticks_minor

            # Scale x-axis limits
            self.xlim = tuple(x * E0 for x in self.xlim)

            # Scale y-axis limits
            self.ylim = tuple(y / np.pi / E0 for y in self.ylim)

            # Scale x-axis ticks and minor ticks
            axes.xticks = tuple(x * E0 for x in axes.xticks)
            axes.xticks_minor = tuple(x * E0 for x in axes.xticks_minor)

            # Scale y-axis ticks and minor ticks
            axes.yticks = tuple(y / np.pi / E0 for y in axes.yticks)
            axes.yticks_minor = tuple(y / np.pi / E0 for y in axes.yticks_minor)

    def plot(self, path: str | Path) -> None:
        """Generate spectral density plot."""

        # Set derived attributes
        self.set_derived_attributes()

        # Unpack and alias histogram data based on unfolding
        if self.unfold:
            bins = self.data.unf_bins
            hist = self.data.unf_hist
        else:
            bins = self.data.bins
            hist = self.data.hist

        # Alias ensemble and ensemble dimension
        ensemble = self.ensemble
        dim = ensemble.dim

        # =================================================

        # Create figure and axes
        self.create_figure()

        # Plot histogram
        self.ax.hist(
            bins[:-1],
            bins=bins,
            weights=hist,
            color=self.hist_color,
            alpha=self.hist_alpha,
            zorder=self.hist_zorder,
        )

        # Create array of energy values for theoretical curve
        energies = np.linspace(self.xlim[0], self.xlim[1], self.num_points)

        # Calculate spectral density based on unfolding
        if self.unfold:
            spec_pdf = np.zeros(self.num_points)
            spec_pdf[np.abs(energies) < dim / 2] = 1 / dim
        else:
            spec_pdf = ensemble.pdf(energies)

        # Plot theoretical spectral density
        self.ax.plot(
            energies,
            spec_pdf,
            color=self.spec_color,
            alpha=self.spec_alpha,
            linewidth=self.spec_width,
            zorder=self.spec_zorder,
        )

        # Finalize and save figure
        self.finish_plot(path=path)
