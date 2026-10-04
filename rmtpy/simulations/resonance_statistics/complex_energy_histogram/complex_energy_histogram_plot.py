import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import cast, override

import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter

from ....compounds import CompoundEnsemble
from ...base_data import Data
from ...base_plot import (
    ENSEMBLE_AVERAGED_CURVE_WIDTH,
    UNFOLDING_LABELS_BY_TYPE,
    ConfigurableAxes,
    Plot,
    PlotAxes,
    PlotLegend,
    format_coupling_label,
)
from .complex_energy_histogram_data import ComplexEnergyHistogram


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class ComplexEnergyHistogramAxes(PlotAxes):
    xticks: tuple[float, ...] = (-1.0, 0.0, 1.0)
    xticks_minor: tuple[float, ...] = (-0.5, 0.5)
    xlabel: str = r"$E / E_0$"
    xtick_labels: tuple[str, ...] = (
        r"$-1.0$",
        r"$0.0$",
        r"$+1.0$",
    )

    yticks: tuple[float, ...] = tuple(range(-4, 5, 2))  # log scale base 10
    yticks_minor: tuple[float, ...] = tuple()
    ylabel: str = r"$\log_{10}(\Gamma / E_0)$"
    ytick_labels: tuple[str, ...] = (
        r"$-4.0$",
        r"$-2.0$",
        r"$0.0$",
        r"$+2.0$",
        r"$+4.0$",
    )

    @override
    def configure(self, axes: ConfigurableAxes) -> None:
        super().configure(axes)
        _ = axes.tick_params(
            axis="x",
            which="minor",
            length=self.tick_length / 2,
        )


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class ComplexEnergyHistogramPlot(Plot):
    data: Data
    axes: PlotAxes = dataclasses.field(default_factory=ComplexEnergyHistogramAxes)
    num_points: int = 1000

    xlim: tuple[float, float] = (-1.2, 1.2)
    ylim: tuple[float, float] = (-5.0, 5.0)  # log scale base 10

    histogram_zorder: int = 1
    histogram_alpha: float = 1.0
    histogram_color: str = "OrangeRed"
    histogram_legend: str = "simulation"

    width_curve_zorder: int = 2
    width_curve_alpha: float = 1.0
    width_curve_width: float = ENSEMBLE_AVERAGED_CURVE_WIDTH
    width_ENSEMBLE_AVERAGED_CURVE_WIDTH: float = ENSEMBLE_AVERAGED_CURVE_WIDTH
    width_curve_color: str = "Cyan"
    width_curve_legend: str = r"$\ensavg{\Gamma(E)}$"

    legend_labels: tuple[str, str] = (histogram_legend, width_curve_legend)
    legend_handles: tuple[Patch, Line2D] = (
        Patch(color=histogram_color, alpha=histogram_alpha),
        Line2D(
            [0],
            [0],
            color=width_curve_color,
            alpha=width_curve_alpha,
            linewidth=width_ENSEMBLE_AVERAGED_CURVE_WIDTH,
        ),
    )

    _curve_width_aliases_are_set: bool = dataclasses.field(
        default=False,
        init=False,
        repr=False,
    )

    @override
    def __new__[PlotType: ComplexEnergyHistogramPlot](
        cls: type[PlotType], **arguments: object
    ) -> PlotType:
        if (
            "width_curve_width" in arguments
            and "width_ENSEMBLE_AVERAGED_CURVE_WIDTH" in arguments
            and arguments["width_curve_width"]
            != arguments["width_ENSEMBLE_AVERAGED_CURVE_WIDTH"]
        ):
            raise ValueError("Conflicting values for the width-curve aliases.")

        return object.__new__(cls)

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        width = (
            self.width_ENSEMBLE_AVERAGED_CURVE_WIDTH
            if self.width_curve_width == ENSEMBLE_AVERAGED_CURVE_WIDTH
            else self.width_curve_width
        )
        self.width_curve_width = width
        self.width_ENSEMBLE_AVERAGED_CURVE_WIDTH = width
        self._curve_width_aliases_are_set = True

    @override
    def __setattr__(self, name: str, value: object) -> None:
        object.__setattr__(self, name, value)
        if name in {"width_curve_width", "width_ENSEMBLE_AVERAGED_CURVE_WIDTH"} and cast(
            bool, getattr(self, "_curve_width_aliases_are_set", False)
        ):
            alias = (
                "width_ENSEMBLE_AVERAGED_CURVE_WIDTH"
                if name == "width_curve_width"
                else "width_curve_width"
            )
            object.__setattr__(self, alias, value)

    def set_derived_attributes(self) -> None:
        if self._derived_attributes_are_set:
            return

        self.compound: CompoundEnsemble = self.store_manifest_arg(
            "compound", CompoundEnsemble
        )
        ensemble = self.compound.ensemble

        self.legend: PlotLegend = PlotLegend(
            handles=self.legend_handles,
            labels=self.legend_labels,
            on_black_background=True,
            loc="upper right",
            bbox=(0.98, 0.95),
        )

        coupling_label = format_coupling_label(self.compound)

        self.axes.title = (
            "Pole Distribution: "
            + ensemble.to_latex
            + rf", $N_\text{{f}} = {{{self.compound.num_free_complex_fermions}}}$"
            + f", {{{coupling_label}}}"
        )

        self.scale_limits_and_ticks(y=lambda value: 10**value)

        self._derived_attributes_are_set: bool = True

    @override
    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        if not isinstance(self.data, ComplexEnergyHistogram):
            raise ValueError("Data must be a `ComplexEnergyHistogram` instance")

        self.build_figure()

        set_xscale = cast(Callable[..., object], self.ax.set_xscale)
        _ = set_xscale("linear")

        set_yscale = cast(Callable[..., object], self.ax.set_yscale)
        _ = set_yscale("log", base=10)

        _ = self.ax.yaxis.set_minor_formatter(NullFormatter())

        _ = self.ax.set_facecolor("Black")

        tick_params = cast(Callable[..., object], self.ax.tick_params)
        _ = tick_params(axis="both", which="both", color="White")

        histogram = self.data.histogram.copy()
        positive_values = histogram[histogram > 0.0]
        if positive_values.size:
            histogram[histogram == 0.0] = np.nan
            color_min = float(np.min(positive_values))
            color_max = float(np.max(positive_values))
            if color_min == color_max:
                color_max = cast(np.float64, np.nextafter(color_max, np.inf))

            x_mesh, y_mesh = np.meshgrid(
                self.data.x_bins,
                self.data.y_bins,
                indexing="ij",
            )
            pcolormesh = cast(Callable[..., object], self.ax.pcolormesh)
            _ = pcolormesh(
                x_mesh,
                y_mesh,
                histogram,
                shading="flat",
                cmap="magma",
                norm=LogNorm(vmin=color_min, vmax=color_max),
                alpha=self.histogram_alpha,
                zorder=self.histogram_zorder,
            )

            resonance_centers, average_width_given_center = (
                self.data.compute_average_x_curve()
            )
            plot = cast(Callable[..., object], self.ax.plot)
            _ = plot(
                resonance_centers,
                average_width_given_center,
                color=self.width_curve_color,
                alpha=self.width_curve_alpha,
                linewidth=self.width_curve_width,
                zorder=self.width_curve_zorder,
            )

        self.finish_plot(path=path)


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class UnfoldedComplexEnergyHistogramAxes(ComplexEnergyHistogramAxes):
    ylabel: str = r"$\log_{10}\gamma$"


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class UnfoldedComplexEnergyHistogramPlot(ComplexEnergyHistogramPlot):
    axes: PlotAxes = dataclasses.field(default_factory=UnfoldedComplexEnergyHistogramAxes)

    @override
    def set_derived_attributes(self) -> None:
        if self._derived_attributes_are_set:
            return

        self.compound: CompoundEnsemble = self.store_manifest_arg(
            "compound", CompoundEnsemble
        )
        ensemble = self.compound.ensemble

        self.legend: PlotLegend = PlotLegend(
            handles=self.legend_handles,
            labels=self.legend_labels,
            on_black_background=True,
            loc="upper right",
            bbox=(0.98, 0.95),
        )

        coupling_label = format_coupling_label(self.compound)

        unfolding_type = cast(str, self.data.metadata["unfolding"])
        unfolding_label = UNFOLDING_LABELS_BY_TYPE[unfolding_type]
        if unfolding_type != "weight":
            unfolding_degree = self.data.metadata["polynomial_degree"]
            title = f"{unfolding_label}({unfolding_degree})-unfolded"
        else:
            title = f"{unfolding_label}-unfolded"
        self.axes.title = (
            f"{title} Pole Distribution: {ensemble.to_latex}"
            + rf", $N_\text{{f}} = {{{self.compound.num_free_complex_fermions}}}$"
            + f", {{{coupling_label}}}"
        )

        self.scale_limits_and_ticks(y=lambda value: 10**value)

        self._derived_attributes_are_set: bool = True
