import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import cast, override

from matplotlib.lines import Line2D

from ....compounds import CompoundEnsemble
from ...base_data import Data
from ...base_plot import Plot, PlotAxes, PlotLegend
from .transmission_coefficients_data import TransmissionCoefficientsData


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class TransmissionCoefficientsAxes(PlotAxes):
    xticks: tuple[float, ...] = (-1.0, 0.0, 1.0)  # units of energy_0
    xticks_minor: tuple[float, ...] = (-1.5, -0.5, 0.5, 1.5)
    xlabel: str = r"$E / E_0$"
    xtick_labels: tuple[str, ...] = (
        r"$-1.0$",
        r"$0.0$",
        r"$+1.0$",
    )
    ylabel: str = r"$T_a(E)$"


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class TransmissionCoefficientsPlot(Plot):
    data: Data
    axes: PlotAxes = dataclasses.field(default_factory=TransmissionCoefficientsAxes)

    xlim: tuple[float, float] = (-1.8, 1.8)
    ylim: tuple[float, float] = (0.0, 1.2)

    line_zorder: int = 2
    line_width: float = 1.5
    line_alpha: float = 1.0
    line_color: str = "#7b2d26"
    line_legend: str = "simulation"

    legend_labels: tuple[str] = (line_legend,)
    legend_handles: tuple[Line2D] = (
        Line2D(
            [0],
            [0],
            color=line_color,
            alpha=line_alpha,
            linewidth=line_width,
        ),
    )

    def set_derived_attributes(self) -> None:
        self.compound: CompoundEnsemble = self.store_manifest_arg(
            "compound", CompoundEnsemble
        )
        ensemble = self.compound.ensemble
        channel_index = self.data.metadata["channel_index"]

        self.xlim = ensemble.spectral_density.plot_range
        self.axes.xticks = tuple(
            value * ensemble.spectral_radius for value in self.axes.xticks
        )
        self.axes.xticks_minor = tuple(
            value * ensemble.spectral_radius for value in self.axes.xticks_minor
        )
        self.axes.ylabel = rf"$T_{{{channel_index}}}(E)$"

        self.legend: PlotLegend = PlotLegend(
            handles=self.legend_handles,
            labels=self.legend_labels,
            loc="upper right",
            bbox=(0.98, 0.95),
        )
        if not self.legend.title:
            self.legend.title = (
                ensemble.to_latex
                + "\n"
                + rf"$N_\text{{f}} = {{{self.compound.num_free_complex_fermions}}}$"
                + rf", $a = {{{channel_index}}}$"
            )

    @override
    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        if not isinstance(self.data, TransmissionCoefficientsData):
            raise ValueError("Data must be a `TransmissionCoefficientsData` instance")

        self.build_figure()

        plot = cast(Callable[..., object], self.ax.plot)
        _ = plot(
            self.data.energies,
            self.data.transmission_coefficients,
            color=self.line_color,
            alpha=self.line_alpha,
            linewidth=self.line_width,
            zorder=self.line_zorder,
        )

        self.finish_plot(path=path)
