# rmtpy/plotting/spectral_statistics_plots/form_factors_plot/form_factors_legend.py

# Postponed evaluation of annotations
from __future__ import annotations

# Standard library imports
from dataclasses import dataclass

# Local application imports
from ..._base import PlotLegend


# -----------------------------
# Form Factors Legend Dataclass
# -----------------------------
@dataclass(repr=False, eq=False, kw_only=True)
class FormFactorsLegend(PlotLegend):

    # Legend location
    loc: str = "upper right"

    # Bounding box anchor
    bbox: tuple[float, float] = (0.735, 0.9)

    # Unfolded bounding box anchor
    unf_bbox: tuple[float, float] = (0.76, 0.96)
