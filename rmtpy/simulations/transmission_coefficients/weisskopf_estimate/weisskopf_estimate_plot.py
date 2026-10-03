import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import cast, override

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

from ....compounds import CompoundEnsemble
from ...base_data import Data
from ...base_plot import ENSEMBLE_AVERAGED_CURVE_WIDTH, Plot, PlotAxes, PlotLegend
from .weisskopf_estimate_data import WeisskopfEstimateData


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
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


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class WeisskopfEstimatePlot(Plot):
    data: Data
    axes: PlotAxes = dataclasses.field(default_factory=WeisskopfEstimateAxes)

    xlim: tuple[float, float] = (-1.0, 1.0)

    line_zorder: int = 2
    line_width: float = ENSEMBLE_AVERAGED_CURVE_WIDTH
    line_alpha: float = 1.0
    line_color: str = "#28536b"

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
        mean_coupling_squared = float(np.mean(self.compound.couplings**2))
        ensemble = self.compound.ensemble

        self.xlim = ensemble.spectral_density.plot_range
        self.axes.xticks = tuple(
            value * ensemble.spectral_radius for value in self.axes.xticks
        )
        self.axes.xticks_minor = tuple(
            value * ensemble.spectral_radius for value in self.axes.xticks_minor
        )

        coupling_exponent = cast(
            float, np.log10(mean_coupling_squared / ensemble.spectral_radius)
        )
        coupling_exponent = 0.0 if abs(coupling_exponent) < 0.005 else coupling_exponent
        coupling_label = rf"$\alpha = {{{coupling_exponent:.1f}}}$"

        self.legend: PlotLegend = PlotLegend(
            handles=self.legend_handles,
            labels=(coupling_label,),
            loc="upper right",
            bbox=(0.74, 0.95),
        )
        self.axes.title = (
            "Weisskopf Estimate: "
            + ensemble.to_latex
            + rf", $N_\text{{f}} = {{{self.compound.num_free_complex_fermions}}}$"
        )

    @override
    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        if not isinstance(self.data, WeisskopfEstimateData):
            raise ValueError("Data must be a `WeisskopfEstimateData` instance")

        self.build_figure()

        self.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        plot = cast(Callable[..., object], self.ax.plot)
        _ = plot(
            self.data.energies,
            self.data.weisskopf_estimate,
            color=self.line_color,
            alpha=self.line_alpha,
            linewidth=self.line_width,
            zorder=self.line_zorder,
        )

        self.finish_plot(path=path)
