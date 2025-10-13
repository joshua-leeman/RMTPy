# rmtpy/plotting/spectral_statistics/spectral_density/spectral_density_legend.py

# Postponed evaluation of annotations
from __future__ import annotations

# Standard library imports
from dataclasses import dataclass

# Local application imports
from ...base import PlotLegend


# ---------------------------------
# Spectral Density Legend Dataclass
# ---------------------------------
@dataclass(repr=False, eq=False, kw_only=True)
class SpectralDensityLegend(PlotLegend):
    # Legend location
    loc: str = "upper right"

    # Bounding box anchor
    bbox: tuple[float, float] = (0.94, 0.95)
