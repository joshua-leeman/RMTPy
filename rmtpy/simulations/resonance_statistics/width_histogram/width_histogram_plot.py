import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import cast, override

import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter

from ....compounds import CompoundEnsemble
from ...base_data import Data
from ...base_plot import Plot, PlotAxes, PlotLegend
from .width_histogram_data import WidthHistogram


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class WidthHistogramAxes(PlotAxes):
    xticks: tuple[float, ...] = tuple(range(-3, 4))  # log scale base 10
    xticks_minor: tuple[float, ...] = tuple()
    xlabel: str = r"$x = \log_{10} (\Gamma / E_0)$"
    xtick_labels: tuple[str, ...] = (
        r"$-3.0$",
        r"$-2.0$",
        r"$-1.0$",
        r"$0.0$",
        r"$+1.0$",
        r"$+2.0$",
        r"$+3.0$",
    )

    yticks: tuple[float, ...] = tuple(range(-3, 4))  # log scale base 10
    yticks_minor: tuple[float, ...] = tuple()
    ylabel: str = r"$P(x)$"
    ytick_labels: tuple[str, ...] = (
        r"$-3.0$",
        r"$-2.0$",
        r"$-1.0$",
        r"$0.0$",
        r"$+1.0$",
        r"$+2.0$",
        r"$+3.0$",
    )


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class WidthHistogramPlot(Plot):
    data: Data
    axes: PlotAxes = dataclasses.field(default_factory=WidthHistogramAxes)

    xlim: tuple[float, float] = (-3.5, 3.5)  # log scale base 10
    ylim: tuple[float, float] = (-3.5, 3.5)  # log scale base 10

    histogram_zorder: int = 1
    histogram_alpha: float = 0.5
    histogram_color: str = "SeaGreen"
    histogram_legend: str = "simulation"

    legend_labels: tuple[str] = (histogram_legend,)
    legend_handles: tuple[Patch] = (Patch(color=histogram_color, alpha=histogram_alpha),)

    def set_derived_attributes(self) -> None:
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
        self.axes.title = (
            "Resonance Width Distribution: "
            + ensemble.to_latex
            + rf", $N_\text{{f}} = {{{self.compound.num_free_complex_fermions}}}$"
            + f", {{{coupling_label}}}"
        )

        self.scale_limits_and_ticks(
            x=lambda value: 10**value,
            y=lambda value: 10**value,
        )

    @override
    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        if not isinstance(self.data, WidthHistogram):
            raise ValueError("Data must be a `WidthHistogram` instance")

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

        self.finish_plot(path=path)


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class UnfoldedWidthHistogramAxes(PlotAxes):
    xticks: tuple[float, ...] = tuple(range(-3, 4))  # log scale base 10
    xlabel: str = r"$\zeta = \log_{10} \gamma$"
    xtick_labels: tuple[str, ...] = (
        r"$-3.0$",
        r"$-2.0$",
        r"$-1.0$",
        r"$0.0$",
        r"$+1.0$",
        r"$+2.0$",
        r"$+3.0$",
    )

    yticks: tuple[float, ...] = tuple(range(-3, 4))  # log scale base 10
    ylabel: str = r"$P(\zeta)$"
    ytick_labels: tuple[str, ...] = (
        r"$-3.0$",
        r"$-2.0$",
        r"$-1.0$",
        r"$0.0$",
        r"$+1.0$",
        r"$+2.0$",
        r"$+3.0$",
    )


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class UnfoldedWidthHistogramPlot(Plot):
    data: Data
    axes: PlotAxes = dataclasses.field(default_factory=UnfoldedWidthHistogramAxes)

    xlim: tuple[float, float] = (-3.5, 3.5)  # log scale base 10
    ylim: tuple[float, float] = (-3.5, 3.5)  # log scale base 10

    histogram_zorder: int = 1
    histogram_alpha: float = 0.5
    histogram_color: str = "SeaGreen"
    histogram_legend: str = "simulation"

    legend_labels: tuple[str] = (histogram_legend,)
    legend_handles: tuple[Patch] = (Patch(color=histogram_color, alpha=histogram_alpha),)

    def set_derived_attributes(self) -> None:
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
        unfolding_type = cast(str, self.data.metadata["unfolding"])
        unfolding_label = (
            "Average" if unfolding_type == "averaged" else unfolding_type.capitalize()
        )
        title = f"{unfolding_label}-unfolded"
        if unfolding_type != "weight":
            unfolding_degree = self.data.metadata["polynomial_degree"]
            title += f" (deg = ${unfolding_degree}$)"
        self.axes.title = (
            f"{title} Resonance Width Distribution: {ensemble.to_latex}"
            + rf", $N_\text{{f}} = {{{self.compound.num_free_complex_fermions}}}$"
            + f", {{{coupling_label}}}"
        )

        self.scale_limits_and_ticks(
            x=lambda value: 10**value,
            y=lambda value: 10**value,
        )

    @override
    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        if not isinstance(self.data, WidthHistogram):
            raise ValueError("Data must be a `WidthHistogram` instance")

        self.build_figure()

        set_xscale = cast(Callable[..., object], self.ax.set_xscale)
        _ = set_xscale("log", base=10)

        set_yscale = cast(Callable[..., object], self.ax.set_yscale)
        _ = set_yscale("log", base=10)

        self.draw_histogram(
            color=self.histogram_color,
            alpha=self.histogram_alpha,
            zorder=self.histogram_zorder,
        )

        self.finish_plot(path=path)
