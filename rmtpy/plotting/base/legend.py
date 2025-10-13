# rmtpy/plotting/base/legend.py

# Postponed evaluation of annotations
from __future__ import annotations

# Standard library imports
from dataclasses import dataclass

# Third-party imports
from matplotlib.axes import Axes


# ----------------------
# Plot Legend Base Class
# ----------------------
@dataclass(repr=False, eq=False, kw_only=True)
class PlotLegend:
    # Legend handles
    handles: tuple | None = None

    # Legend labels
    labels: tuple[str, ...] | None = None

    # Legend text properties
    fontsize: int = 10
    textalignment: str = "left"

    # Legend title
    title: str | None = None
    title_fontsize: int = 10

    # Legend location
    loc: str = "best"
    bbox: tuple[float, float] | None = None

    # Legend frame
    frameon: bool = False

    def configure(self, ax: Axes) -> None:
        """Configure legend with specified settings."""
        # Set legend if handles and labels are provided
        if self.handles is not None and self.labels is not None:
            ax.legend(
                handles=self.handles,
                labels=self.labels,
                title=self.title,
                loc=self.loc,
                bbox_to_anchor=self.bbox,
                frameon=self.frameon,
                fontsize=self.fontsize,
                title_fontsize=self.title_fontsize,
                alignment=self.textalignment,
            )
