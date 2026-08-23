from __future__ import annotations

import dataclasses
from pathlib import Path

from matplotlib.lines import Line2D

from rmtpy.compounds import Compound

from ...plot import Plot, PlotAxes, PlotLegend
from .transmission_coefficients_data import TransmissionCoefficientsData


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class TransmissionCoefficientsAxes(PlotAxes):
    xticks: tuple[float, ...] = (-1.0, 0.0, 1.0)  # units of energy_0
    xticks_minor: tuple[float, ...] = (-0.5, 0.5)
    xlabel: str = r"$E / E_0$"
    xtick_labels: tuple[str, ...] = (
        r"$-1.0$",
        r"$0.0$",
        r"$+1.0$",
    )
    ylabel: str = r"$T_a(E)$"


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class TransmissionCoefficientsPlot(Plot):
    """Plot one channel's transmission coefficient across energy."""

    data: TransmissionCoefficientsData
    axes: TransmissionCoefficientsAxes = dataclasses.field(
        default_factory=TransmissionCoefficientsAxes
    )

    line_color: str = "#7b2d26"
    line_width: float = 1.5
    line_legend: str = "simulation"

    def set_derived_attributes(self) -> None:
        compound = self.structure_simulation_arg("compound", Compound)
        energy_scale = compound.ensemble.spectral_radius
        channel_index = self.data.channel_index

        self.xlim = compound.ensemble.spectral_density.plot_range
        self.axes.xticks = tuple(value * energy_scale for value in self.axes.xticks)
        self.axes.xticks_minor = tuple(
            value * energy_scale for value in self.axes.xticks_minor
        )
        self.axes.ylabel = rf"$T_{{{channel_index}}}(E)$"
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
                + rf", $a = {{{channel_index}}}$"
            ),
            loc="upper right",
            bbox=(0.98, 0.95),
        )

    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()
        self.create_figure()

        self.ax.plot(
            self.data.energies,
            self.data.transmission_coefficients,
            color=self.line_color,
            linewidth=self.line_width,
        )

        self.ylim = (-0.02, 1.02)

        self.finish_plot(path=path)
