import dataclasses
from pathlib import Path

from matplotlib.patches import Patch

from ....compounds import Compound
from ...histogram import Histogram
from ...plot import Plot, PlotAxes, PlotLegend


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class ResonanceCoefficientHistogramLegend(PlotLegend):
    loc: str = "upper right"
    bbox: tuple[float, float] = (0.94, 0.95)


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class ResonanceCoefficientHistogramAxes(PlotAxes):
    xticks: tuple[float, ...] = (-0.2, -0.1, 0.0, 0.1, 0.2)
    xticks_minor: tuple[float, ...] = (-0.15, -0.05, 0.05, 0.15)
    xlabel: str = r"$c$"
    xtick_labels: tuple[str, ...] = (
        r"$-0.2$",
        r"$-0.1$",
        r"$0.0$",
        r"$+0.1$",
        r"$+0.2$",
    )

    yticks: tuple[float, ...] = tuple(range(0, 24, 4))
    yticks_minor: tuple[float, ...] = tuple(range(2, 22, 4))
    ylabel: str = r"$\ensavg{\rho(c)}$"
    ytick_labels: tuple[str, ...] = (
        r"$0$",
        r"$4$",
        r"$8$",
        r"$12$",
        r"$16$",
        r"$20$",
    )


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class ResonanceCoefficientHistogramPlot(Plot):
    data: Histogram
    axes: ResonanceCoefficientHistogramAxes = dataclasses.field(
        default_factory=ResonanceCoefficientHistogramAxes
    )

    xlim: tuple[float, float] = (-0.2, 0.2)
    ylim: tuple[float, float] = (0.0, 20.0)

    histogram_zorder: int = 1
    histogram_alpha: float = 0.5
    histogram_color: str = "OrangeRed"
    histogram_legend: str = "simulation"

    legend_labels: tuple[str, ...] = (histogram_legend,)
    legend_handles: tuple[Patch, ...] = (
        Patch(color=histogram_color, alpha=histogram_alpha),
    )

    def set_derived_attributes(self) -> None:
        self.compound: Compound = self.structure_simulation_arg("compound", Compound)

        self.legend: ResonanceCoefficientHistogramLegend = (
            ResonanceCoefficientHistogramLegend(
                handles=self.legend_handles, labels=self.legend_labels
            )
        )
        if self.legend.title is None:
            self.legend.title = self.compound.to_latex

    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        self.create_figure()

        self.draw_histogram(
            color=self.histogram_color,
            alpha=self.histogram_alpha,
            zorder=self.histogram_zorder,
        )

        self.finish_plot(path=path)
