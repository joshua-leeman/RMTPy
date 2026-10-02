import dataclasses
from pathlib import Path
from typing import cast, override

import numpy as np
from matplotlib.patches import Patch

from ....compounds import CompoundEnsemble
from ...base_data import Data
from ...base_plot import Plot, PlotAxes, PlotLegend
from .resonance_coefficients_histogram_data import ResonanceCoefficientsHistogram


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class ResonanceCoefficientsHistogramAxes(PlotAxes):
    xticks: tuple[float, ...] = (-0.2, -0.1, 0.0, 0.1, 0.2)
    xticks_minor: tuple[float, ...] = (-0.15, -0.05, 0.05, 0.15)
    xtick_labels: tuple[str, ...] = (
        r"$-0.2$",
        r"$-0.1$",
        r"$0.0$",
        r"$+0.1$",
        r"$+0.2$",
    )

    yticks: tuple[float, ...] = tuple(range(0, 24, 4))
    yticks_minor: tuple[float, ...] = tuple(range(2, 22, 4))
    ytick_labels: tuple[str, ...] = (
        r"$0$",
        r"$4$",
        r"$8$",
        r"$12$",
        r"$16$",
        r"$20$",
    )


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class ResonanceCoefficientsHistogramPlot(Plot):
    data: Data
    axes: PlotAxes = dataclasses.field(
        default_factory=ResonanceCoefficientsHistogramAxes,
    )

    xlim: tuple[float, float] = (-0.5, 0.5)
    ylim: tuple[float, float] = (0.0, 50.0)

    histogram_zorder: int = 1
    histogram_alpha: float = 0.5
    histogram_color: str = "OrangeRed"
    histogram_legend: str = "simulation"

    legend_labels: tuple[str] = (histogram_legend,)
    legend_handles: tuple[Patch] = (Patch(color=histogram_color, alpha=histogram_alpha),)

    def set_derived_attributes(self) -> None:
        coefficient_degree = cast(int, self.data.metadata["degree"])
        self.axes.xlabel = rf"$c_{{{coefficient_degree}}}$"
        self.axes.ylabel = rf"$P(c_{{{coefficient_degree}}})$"

        self.compound: CompoundEnsemble = self.store_manifest_arg(
            "compound", CompoundEnsemble
        )
        mean_coupling_squared = float(np.mean(self.compound.couplings**2))
        ensemble = self.compound.ensemble

        self.legend: PlotLegend = PlotLegend(
            handles=self.legend_handles,
            labels=self.legend_labels,
            loc="upper right",
            bbox=(0.94, 0.95),
        )

        coupling_exponent = cast(
            float, np.log10(mean_coupling_squared / ensemble.spectral_radius)
        )
        coupling_exponent = 0.0 if abs(coupling_exponent) < 0.005 else coupling_exponent
        coupling_label = rf"$\alpha = {{{coupling_exponent:.1f}}}$"
        if not self.legend.title:
            self.legend.title = (
                ensemble.to_latex
                + "\n"
                + rf"$N_\text{{f}} = {{{self.compound.num_free_complex_fermions}}}$"
                + f", {{{coupling_label}}}"
            )

    @override
    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        if not isinstance(self.data, ResonanceCoefficientsHistogram):
            raise ValueError("Data must be a `ResonanceCoefficientsHistogram` instance")

        self.build_figure()

        self.draw_histogram(
            color=self.histogram_color,
            alpha=self.histogram_alpha,
            zorder=self.histogram_zorder,
        )

        self.finish_plot(path=path)
