import dataclasses
from pathlib import Path
from typing import cast, override

from matplotlib.patches import Patch

from ....ensembles import ManyBodyEnsemble
from ...base_data import Data
from ...base_plot import Plot, PlotAxes, PlotLegend


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class SpectralCoefficientsHistogramAxes(PlotAxes):
    xticks: tuple[float, ...] = (-0.2, -0.1, 0.0, 0.1, 0.2)  # units of energy_0
    xticks_minor: tuple[float, ...] = (-0.15, -0.05, 0.05, 0.15)
    xtick_labels: tuple[str, ...] = (
        r"$-0.2$",
        r"$-0.1$",
        r"$0.0$",
        r"$+0.1$",
        r"$+0.2$",
    )

    yticks: tuple[float, ...] = tuple(range(0, 24, 4))  # units of 1 / (pi * energy_0)
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
class SpectralCoefficientsHistogramPlot(Plot):
    data: Data
    axes: PlotAxes = dataclasses.field(
        default_factory=SpectralCoefficientsHistogramAxes,
    )

    xlim: tuple[float, float] = (-0.25, 0.25)  # units of energy_0
    ylim: tuple[float, float] = (0.0, 20)  # units of 1 / (pi * energy_0)

    histogram_zorder: int = 1
    histogram_alpha: float = 0.5
    histogram_color: str = "RoyalBlue"
    histogram_legend: str = "simulation"

    legend_labels: tuple[str] = (histogram_legend,)
    legend_handles: tuple[Patch] = (Patch(color=histogram_color, alpha=histogram_alpha),)

    def set_derived_attributes(self) -> None:
        coeff_degree = cast(int, self.data.metadata["degree"])
        self.axes.xlabel = rf"$c_{{{coeff_degree}}}$"
        self.axes.ylabel = rf"$P(c_{{{coeff_degree}}})$"

        self.ensemble: ManyBodyEnsemble = self.store_manifest_arg(
            "ensemble", ManyBodyEnsemble
        )

        self.legend: PlotLegend = PlotLegend(
            handles=self.legend_handles,
            labels=self.legend_labels,
            loc="upper right",
            bbox=(0.94, 0.95),
        )

        if not self.legend.title:
            self.legend.title = self.ensemble.to_latex

    @override
    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        self.build_figure()

        self.draw_histogram(
            color=self.histogram_color,
            alpha=self.histogram_alpha,
            zorder=self.histogram_zorder,
        )

        self.finish_plot(path=path)
