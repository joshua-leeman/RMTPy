# rmtpy/plotting/spectral_statistics_plots/spacing_histogram_plot/spacing_histogram_legend.py

# Postponed evaluation of annotations
from __future__ import annotations

# Standard library imports
from dataclasses import dataclass

# Local application imports
from ..._base import PlotLegend


# ----------------------------------
# Spacing Histogram Legend Dataclass
# ----------------------------------
@dataclass(repr=False, eq=False, kw_only=True)
class SpacingHistogramLegend(PlotLegend):

    # Legend location
    loc: str = "upper right"

    # Bounding box anchor
    bbox: tuple[float, float] = (0.94, 0.95)
