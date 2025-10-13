# rmtpy/plotting/base/axes.py

# Postponed evaluation of annotations
from __future__ import annotations

# Standard library imports
from abc import ABC
from dataclasses import dataclass

# Third-party imports
from matplotlib.axes import Axes


# --------------------
# Plot Axes Base Class
# --------------------
@dataclass(repr=False, eq=False, kw_only=True)
class PlotAxes(ABC):
    # Axes width
    axes_width: float = 1.0

    # x-axis label
    xlabel: str = r"$x$"
    xlabel_fontsize: int = 12

    # y-axis label
    ylabel: str = r"$y$"
    ylabel_fontsize: int = 12

    # x-axis ticks
    xticks: tuple[float, ...] | None = None
    xticks_minor: tuple[float, ...] | None = None

    # y-axis ticks
    yticks: tuple[float, ...] | None = None
    yticks_minor: tuple[float, ...] | None = None

    # Tick length
    tick_length: float = 6.0

    # Axes tick labels
    xtick_labels: tuple[str, ...] | None = None
    ytick_labels: tuple[str, ...] | None = None
    tick_fontsize: int = 10

    def configure(self, ax: Axes) -> None:
        """Configure axes with specified settings."""
        # Set axes line width
        for spine in ax.spines.values():
            spine.set_linewidth(self.axes_width)

        # Set x-axis label
        ax.set_xlabel(self.xlabel, fontsize=self.xlabel_fontsize)

        # Set y-axis label
        ax.set_ylabel(self.ylabel, fontsize=self.ylabel_fontsize)

        # Set x-ticks
        if self.xticks is not None:
            ax.set_xticks(self.xticks)

        # Set minor x-ticks
        if self.xticks_minor is not None:
            ax.set_xticks(self.xticks_minor, minor=True)

        # Set y-ticks
        if self.yticks is not None:
            ax.set_yticks(self.yticks)

        # Set minor y-ticks
        if self.yticks_minor is not None:
            ax.set_yticks(self.yticks_minor, minor=True)

        # Set tick parameters
        ax.tick_params(
            direction="in",
            top=True,
            bottom=True,
            left=True,
            right=True,
            which="both",
            length=self.tick_length,
        )

        # Set x-tick labels
        if self.xtick_labels is not None:
            ax.set_xticklabels(self.xtick_labels, fontsize=self.tick_fontsize)
        else:
            # Change x-tick label font size
            ax.tick_params(axis="x", labelsize=self.tick_fontsize)

        # Set y-tick labels
        if self.ytick_labels is not None:
            ax.set_yticklabels(self.ytick_labels, fontsize=self.tick_fontsize)
        else:
            # Change y-tick label font size
            ax.tick_params(axis="y", labelsize=self.tick_fontsize)
