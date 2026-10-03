import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import cast, override

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter

from ....compounds import CompoundEnsemble
from ....density import compute_bin_centers
from ...base_data import Data
from ...base_plot import CURVE_WIDTH, Plot, PlotAxes, PlotLegend
from .total_width_histogram_data import TotalWidthHistogram


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class TotalWidthHistogramAxes(PlotAxes):
    xticks: tuple[float, ...] = tuple(range(-1, 2))  # log scale base 10
    xticks_minor: tuple[float, ...] = ()
    xlabel: str = r"$\log_{10} Y$"
    xtick_labels: tuple[str, ...] = (
        r"$-1.0$",
        r"$0.0$",
        r"$+1.0$",
    )

    yticks: tuple[float, ...] = tuple(range(-3, 2))  # log scale base 10
    yticks_minor: tuple[float, ...] = ()
    ylabel: str = r"$\log_{10} P(Y)$"
    ytick_labels: tuple[str, ...] = (
        r"$-3.0$",
        r"$-2.0$",
        r"$-1.0$",
        r"$0.0$",
        r"$+1.0$",
    )


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class TotalWidthHistogramPlot(Plot):
    data: Data
    axes: PlotAxes = dataclasses.field(default_factory=TotalWidthHistogramAxes)

    xlim: tuple[float, float] = (-1.2, 1.2)  # log scale base 10
    ylim: tuple[float, float] = (-3.2, 1.2)  # log scale base 10

    histogram_zorder: int = 1
    histogram_alpha: float = 0.5
    histogram_color: str = "BlueViolet"
    histogram_legend: str = "simulation"

    porter_thomas_zorder: int = 2
    porter_thomas_width: float = CURVE_WIDTH
    porter_thomas_alpha: float = 1.0
    porter_thomas_color: str = "Black"
    porter_thomas_legend: str = "Porter-Thomas"

    legend_labels: tuple[str, str] = (histogram_legend, porter_thomas_legend)
    legend_handles: tuple[Patch, Line2D] = (
        Patch(color=histogram_color, alpha=histogram_alpha),
        Line2D(
            [0],
            [0],
            color=porter_thomas_color,
            alpha=porter_thomas_alpha,
            linewidth=porter_thomas_width,
        ),
    )

    def set_derived_attributes(self) -> None:
        width_index = cast(list[int], self.data.metadata["index"])
        state_index = width_index[0]

        self.axes.xlabel = rf"$\log_{{10}} Y_{{{state_index}}}$"
        self.axes.ylabel = rf"$\log_{{10}} P(Y_{{{state_index}}})$"

        self.compound: CompoundEnsemble = self.store_manifest_arg(
            "compound", CompoundEnsemble
        )

        self.legend: PlotLegend = PlotLegend(
            handles=self.legend_handles,
            labels=self.legend_labels,
            loc="upper right",
            bbox=(0.97, 0.95),
        )

        self.axes.title = (
            "Total Widths: "
            + self.compound.ensemble.to_latex
            + rf", $N_\text{{f}} = {{{self.compound.num_free_complex_fermions}}}$"
            + rf", $\mu = {{{state_index}}}$"
        )

        self.scale_limits_and_ticks(
            x=lambda value: 10**value,
            y=lambda value: 10**value,
        )

    @override
    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        if not isinstance(self.data, TotalWidthHistogram):
            raise ValueError("Data must be a `TotalWidthHistogram` instance")

        self.build_figure()

        set_xscale = cast(Callable[..., object], self.ax.set_xscale)
        _ = set_xscale("log", base=10)

        set_yscale = cast(Callable[..., object], self.ax.set_yscale)
        _ = set_yscale("log", base=10)

        _ = self.ax.xaxis.set_minor_formatter(NullFormatter())
        _ = self.ax.yaxis.set_minor_formatter(NullFormatter())

        self.draw_histogram(
            color=self.histogram_color,
            alpha=self.histogram_alpha,
            zorder=self.histogram_zorder,
        )

        widths = compute_bin_centers(self.data.bins)
        porter_thomas_pdf = self.compound.ensemble.porter_thomas_distribution(
            widths,
            num_channels=self.compound.num_channels,
        )

        plot = cast(Callable[..., object], self.ax.plot)
        _ = plot(
            widths,
            porter_thomas_pdf,
            color=self.porter_thomas_color,
            alpha=self.porter_thomas_alpha,
            linewidth=self.porter_thomas_width,
            zorder=self.porter_thomas_zorder,
        )

        self.finish_plot(path=path)
