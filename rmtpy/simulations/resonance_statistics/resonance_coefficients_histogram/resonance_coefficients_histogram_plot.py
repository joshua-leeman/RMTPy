import dataclasses
from pathlib import Path
from typing import cast, override

from matplotlib.patches import Patch

from ....compounds import CompoundEnsemble
from ...base_data import Data
from ...base_plot import (
    Plot,
    PlotAxes,
    PlotLegend,
    configure_coefficient_histogram_axes,
    format_coupling_label,
)
from .resonance_coefficients_histogram_data import ResonanceCoefficientsHistogram


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class ResonanceCoefficientsHistogramAxes(PlotAxes):
    pass


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class ResonanceCoefficientsHistogramPlot(Plot):
    data: Data
    axes: PlotAxes = dataclasses.field(
        default_factory=ResonanceCoefficientsHistogramAxes,
    )

    histogram_zorder: int = 1
    histogram_alpha: float = 0.5
    histogram_color: str = "OrangeRed"
    histogram_legend: str = "simulation"

    legend_labels: tuple[str] = (histogram_legend,)
    legend_handles: tuple[Patch] = (Patch(color=histogram_color, alpha=histogram_alpha),)

    def set_derived_attributes(self) -> None:
        if self._derived_attributes_are_set:
            return

        if not isinstance(self.data, ResonanceCoefficientsHistogram):
            raise ValueError("Data must be a `ResonanceCoefficientsHistogram` instance")

        coefficient_degree = cast(int, self.data.metadata["degree"])
        self.axes.xlabel = rf"$c_{{{coefficient_degree}}}$"
        self.axes.ylabel = rf"$P(c_{{{coefficient_degree}}})$"

        horizontal_limits, vertical_limits = configure_coefficient_histogram_axes(
            self.data, self.axes
        )
        self.xlim: tuple[float, ...] = horizontal_limits
        self.ylim: tuple[float, ...] = vertical_limits

        self.compound: CompoundEnsemble = self.store_manifest_arg(
            "compound", CompoundEnsemble
        )
        ensemble = self.compound.ensemble

        self.legend: PlotLegend = PlotLegend(
            handles=self.legend_handles,
            labels=self.legend_labels,
            loc="upper right",
            bbox=(0.94, 0.95),
        )

        coupling_label = format_coupling_label(self.compound)

        self.axes.title = (
            "Resonance Coefficients: "
            + ensemble.to_latex
            + rf", $N_\text{{f}} = {{{self.compound.num_free_complex_fermions}}}$"
            + f", {{{coupling_label}}}"
        )

        self._derived_attributes_are_set: bool = True

    @override
    def plot(self, path: str | Path) -> None:
        if not self._derived_attributes_are_set:
            self.set_derived_attributes()

        self.build_figure()

        self.draw_histogram(
            color=self.histogram_color,
            alpha=self.histogram_alpha,
            zorder=self.histogram_zorder,
        )

        self.finish_plot(path=path)
