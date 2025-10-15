# rmtpy/plotting/spectral_statistics_plots/form_factors_plot/form_factors_axes.py

# Postponed evaluation of annotations
from __future__ import annotations

# Standard library imports
from dataclasses import dataclass

# Local application imports
from ..._base import PlotAxes


# ---------------------------
# Form Factors Axes Dataclass
# ---------------------------
@dataclass(repr=False, eq=False, kw_only=True)
class FormFactorsAxes(PlotAxes):

    # -----------------
    # x-axis Properties
    # -----------------

    # x-axis labels
    xlabel: str = r"$Jt / j_{1,1}$"
    unf_xlabel: str = r"$\tau$"

    # x-axis ticks (base dim log scale)
    xticks: tuple[float, ...] = (0.0, 0.5, 1.0)  # factor of j_1_1/(E0)

    # unfolded x-axis ticks (base unf log scale)
    unf_xticks: tuple[float, ...] = (-1.0, -0.5, 0.0)  # factor of 2 pi

    # x-axis tick labels
    xtick_labels: tuple[str, ...] = (r"$1/N$", r"$\sqrt{D}/N$", r"$D/N$")
    unf_xtick_labels: tuple[str, ...] = (r"$2\pi / D$", r"$2\pi / \sqrt{D}$", r"$2\pi$")

    # -----------------
    # y-axis Properties
    # -----------------

    # y-axis labels
    ylabel: str = r"$K(t)$"
    unf_ylabel: str = r"$K(\tau)$"

    # y-axis ticks
    yticks: tuple[float, ...] = (-2, -1, 0)  # base dim log scale

    # y-axis tick labels
    ytick_labels: tuple[str, ...] = (r"$D^{-2}$", r"$D^{-1}$", r"$1$")
