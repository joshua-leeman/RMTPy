# rmtpy/plotting/spectral_statistics_plots/spacing_histogram_figure/spacing_histogram_axes.py

# Postponed evaluation of annotations
from __future__ import annotations

# Standard library imports
from dataclasses import dataclass

# Third-party imports
import numpy as np

# Local application imports
from ..._base import PlotAxes


# ---------------------------------
# Spectral Histogram Axes Dataclass
# ---------------------------------
@dataclass(repr=False, eq=False, kw_only=True)
class SpacingHistogramAxes(PlotAxes):

    # -----------------
    # x-axis Properties
    # -----------------

    # x-axis labels
    xlabel: str = r"$\Delta E / d$"
    unf_xlabel: str = r"$s$"

    # x-axis ticks
    xticks: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0)  # factor of d

    # x-axis minor ticks
    xticks_minor: tuple[float, ...] = (0.5, 1.5, 2.5, 3.5)  # factor of d

    # x-axis tick labels
    xtick_labels: tuple[str, ...] = (r"$0$", r"$d$", r"$2d$", r"$3d$", r"$4d$")
    unf_xtick_labels: tuple[str, ...] = (r"$0$", r"$1$", r"$2$", r"$3$", r"$4$")

    # -----------------
    # y-axis Properties
    # -----------------

    # y-axis labels
    ylabel: str = r"$\ensavg{f(\Delta E)}$"
    unf_ylabel: str = r"$\ensavg{f(s)}$"

    # y-axis ticks
    yticks: tuple[float, ...] = (0.0, 0.5, 1.0)  # factor of 1/d

    # y-axis minor ticks
    yticks_minor: tuple[float, ...] = (0.25, 0.75)  # factor of 1/d

    # y-axis tick labels
    ytick_labels: tuple[str, ...] = (r"$0$", r"$\frac{1}{2}d^{-1}$", r"$d^{-1}$")
    unf_ytick_labels: tuple[str, ...] = (r"$0$", r"$\frac{1}{2}$", r"$1$")
