from __future__ import annotations

import dataclasses
from pathlib import Path

from matplotlib.lines import Line2D

from rmtpy.compounds import Compound

from ...plot import Plot, PlotAxes, PlotLegend
from .weisskopf_estimate_data import WeisskopfEstimateData


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class WeisskopfEstimateAxes(PlotAxes):
    xticks: tuple[float, ...] = (-1.0, 0.0, 1.0)  # units of energy_0
    xticks_minor: tuple[float, ...] = (-0.5, 0.5)
    xlabel: str = r"$E / E_0$"
    xtick_labels: tuple[str, ...] = (
        r"$-1.0$",
        r"$0.0$",
        r"$+1.0$",
    )
    ylabel: str = r"$\Gamma_{\mathrm{Weisskopf}}(E)$"


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class WeisskopfEstimatePlot(Plot):
    """Plot the all-channel Weisskopf estimate across energy."""

    data: WeisskopfEstimateData
    axes: WeisskopfEstimateAxes = dataclasses.field(
        default_factory=WeisskopfEstimateAxes
    )

    line_color: str = "#28536b"
    line_width: float = 1.5
    line_legend: str = "Weisskopf estimate"

    def set_derived_attributes(self) -> None:
        compound = self.structure_simulation_arg("compound", Compound)
        energy_scale = compound.ensemble.spectral_radius

        self.xlim = compound.ensemble.spectral_density.plot_range
        self.axes.xticks = tuple(value * energy_scale for value in self.axes.xticks)
        self.axes.xticks_minor = tuple(
            value * energy_scale for value in self.axes.xticks_minor
        )
        self.legend = PlotLegend(
            handles=(
                Line2D(
                    [0],
                    [0],
                    color=self.line_color,
                    linewidth=self.line_width,
                ),
            ),
            labels=(self.line_legend,),
            title=(
                compound.ensemble.to_latex
                + "\n"
                + rf"$N_\text{{f}} = {{{compound.num_free_complex_fermions}}}$"
            ),
            loc="upper right",
            bbox=(0.98, 0.95),
        )

    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()
        self.create_figure()

        self.ax.plot(
            self.data.energies,
            self.data.weisskopf_estimate,
            color=self.line_color,
            linewidth=self.line_width,
        )

        self.finish_plot(path=path)
