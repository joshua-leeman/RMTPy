from __future__ import annotations

import dataclasses
from abc import abstractmethod
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullLocator

import rmtpy.ensembles

from ...plot import (
    DIMENSION_TIME_LOG_SUPPORT,
    DimensionTimeAxes,
    Plot,
    PlotLegend,
)
from .cdo_dynamics_data import CDODynamicsData


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class CDOProbabilityAxes(DimensionTimeAxes):
    ylabel: str = r"$p_n(u)$"
    yticks: tuple[float, ...] = (-2.0, -1.0, 0.0)  # log scale base dimension
    ytick_labels: tuple[str, ...] = (
        r"$D^{-2}$",
        r"$D^{-1}$",
        r"$1$",
    )


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class CDOPurityAxes(DimensionTimeAxes):
    yticks: tuple[float, ...] = (-2.0, -1.0, 0.0)  # log scale base dimension
    ytick_labels: tuple[str, ...] = (
        r"$D^{-2}$",
        r"$D^{-1}$",
        r"$1$",
    )


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class CDOInformationAxes(DimensionTimeAxes):
    yticks: tuple[float, ...] = (0.0, 0.5, 1.0)  # units of log dimension
    ytick_labels: tuple[str, ...] = (
        r"$0$",
        r"$\frac{1}{2}\log D$",
        r"$\log D$",
    )


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class CDOTimePlot(Plot):
    """Common form-factor-style time-axis behavior for CDO views."""

    data: CDODynamicsData
    axes: DimensionTimeAxes = dataclasses.field(default_factory=DimensionTimeAxes)

    xlim: tuple[float, float] = DIMENSION_TIME_LOG_SUPPORT
    num_points: int = 1000

    def set_derived_attributes(self) -> None:
        self.ensemble: rmtpy.ensembles.ManyBodyEnsemble = self.structure_simulation_arg(
            "ensemble",
            rmtpy.ensembles.ManyBodyEnsemble,
        )
        self.scale_limits_and_ticks(
            x=lambda value: self.ensemble.dimension**value * self.data.scale,
        )

    def create_time_figure(self, *, logarithmic_y: bool) -> None:
        self.create_figure()
        dimension = self.ensemble.dimension
        self.ax.set_xscale("log", base=dimension)
        self.ax.xaxis.set_major_locator(
            LogLocator(base=dimension, numticks=len(self.axes.xticks))
        )
        self.ax.xaxis.set_minor_locator(NullLocator())

        if logarithmic_y:
            self.ax.set_yscale("log", base=dimension)
            self.ax.yaxis.set_major_locator(
                LogLocator(base=dimension, numticks=len(self.axes.yticks))
            )
            self.ax.yaxis.set_minor_locator(NullLocator())

    @property
    def positive_time_mask(self) -> np.ndarray:
        return np.isfinite(self.data.times) & (self.data.times > 0.0)

    @abstractmethod
    def plot(self, path: str | Path) -> None:
        raise NotImplementedError()


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class CDOProbabilitiesPlot(CDOTimePlot):
    axes: CDOProbabilityAxes = dataclasses.field(default_factory=CDOProbabilityAxes)
    ylim: tuple[float, float] = (-2.2, 0.2)  # log scale base dimension

    probability_color: str = "#28536b"
    probability_alpha: float = 0.2
    probability_width: float = 0.5
    probability_zorder: int = 2
    probability_legend: str = r"$p_n(u)$"

    equilibrium_color: str = "Black"
    equilibrium_alpha: float = 0.8
    equilibrium_width: float = 0.75
    equilibrium_style: str = "--"
    equilibrium_zorder: int = 1
    equilibrium_legend: str = r"$D^{-1}$"

    @property
    def file_name(self) -> str:
        return "cdo_probabilities_plot"

    def set_derived_attributes(self) -> None:
        super().set_derived_attributes()
        self.legend = PlotLegend(
            handles=(
                Line2D(
                    [0],
                    [0],
                    color=self.probability_color,
                    alpha=self.probability_alpha,
                    linewidth=self.probability_width,
                ),
                Line2D(
                    [0],
                    [0],
                    color=self.equilibrium_color,
                    alpha=self.equilibrium_alpha,
                    linewidth=self.equilibrium_width,
                    linestyle=self.equilibrium_style,
                ),
            ),
            labels=(self.probability_legend, self.equilibrium_legend),
            title=self.ensemble.to_latex,
            loc="upper right",
            bbox=(0.925, 0.95),
        )
        self.scale_limits_and_ticks(y=lambda value: self.ensemble.dimension**value)

    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()
        self.create_time_figure(logarithmic_y=True)

        positive_times = self.data.times[self.positive_time_mask]
        probabilities = self.data.probabilities[self.positive_time_mask]
        self.ax.plot(
            positive_times,
            probabilities,
            color=self.probability_color,
            alpha=self.probability_alpha,
            linewidth=self.probability_width,
            zorder=self.probability_zorder,
        )
        self.ax.axhline(
            1.0 / self.ensemble.dimension,
            color=self.equilibrium_color,
            alpha=self.equilibrium_alpha,
            linewidth=self.equilibrium_width,
            linestyle=self.equilibrium_style,
            zorder=self.equilibrium_zorder,
        )
        self.finish_plot(path=path)


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class CDOPuritiesPlot(CDOTimePlot):
    axes: CDOPurityAxes = dataclasses.field(default_factory=CDOPurityAxes)
    ylim: tuple[float, float] = (-2.2, 0.2)  # log scale base dimension

    classical_color: str = "Blue"
    classical_alpha: float = 1.0
    classical_width: float = 0.75
    classical_zorder: int = 2
    classical_legend: str = r"$\gamma_{\mathrm{c}}(u)$"

    quantum_color: str = "Red"
    quantum_alpha: float = 1.0
    quantum_width: float = 0.75
    quantum_zorder: int = 2
    quantum_legend: str = r"$\gamma_{\mathrm{q}}(u)$"

    @property
    def file_name(self) -> str:
        return "cdo_purities_plot"

    def set_derived_attributes(self) -> None:
        super().set_derived_attributes()
        self.legend = PlotLegend(
            handles=(
                Line2D(
                    [0],
                    [0],
                    color=self.classical_color,
                    alpha=self.classical_alpha,
                    linewidth=self.classical_width,
                ),
                Line2D(
                    [0],
                    [0],
                    color=self.quantum_color,
                    alpha=self.quantum_alpha,
                    linewidth=self.quantum_width,
                ),
            ),
            labels=(self.classical_legend, self.quantum_legend),
            title=self.ensemble.to_latex,
            loc="upper right",
            bbox=(0.925, 0.95),
        )
        self.scale_limits_and_ticks(y=lambda value: self.ensemble.dimension**value)

    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()
        self.create_time_figure(logarithmic_y=True)

        positive_time_mask = self.positive_time_mask
        positive_times = self.data.times[positive_time_mask]
        self.ax.plot(
            positive_times,
            self.data.classical_purity[positive_time_mask],
            color=self.classical_color,
            alpha=self.classical_alpha,
            linewidth=self.classical_width,
            zorder=self.classical_zorder,
        )
        self.ax.plot(
            positive_times,
            self.data.quantum_purity[positive_time_mask],
            color=self.quantum_color,
            alpha=self.quantum_alpha,
            linewidth=self.quantum_width,
            zorder=self.quantum_zorder,
        )
        self.finish_plot(path=path)


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class CDOInformationPlot(CDOTimePlot):
    axes: CDOInformationAxes = dataclasses.field(default_factory=CDOInformationAxes)
    ylim: tuple[float, float] = (-0.02, 1.02)  # units of log dimension

    entropy_color: str = "Blue"
    entropy_alpha: float = 1.0
    entropy_width: float = 0.75
    entropy_zorder: int = 2
    entropy_legend: str = r"$S(u)$"

    kl_color: str = "Red"
    kl_alpha: float = 1.0
    kl_width: float = 0.75
    kl_zorder: int = 2
    kl_legend: str = r"$D_{\mathrm{KL}}(\bar p\Vert p_r)$"

    @property
    def file_name(self) -> str:
        return "cdo_information_plot"

    def set_derived_attributes(self) -> None:
        super().set_derived_attributes()
        self.legend = PlotLegend(
            handles=(
                Line2D(
                    [0],
                    [0],
                    color=self.entropy_color,
                    alpha=self.entropy_alpha,
                    linewidth=self.entropy_width,
                ),
                Line2D(
                    [0],
                    [0],
                    color=self.kl_color,
                    alpha=self.kl_alpha,
                    linewidth=self.kl_width,
                ),
            ),
            labels=(self.entropy_legend, self.kl_legend),
            title=self.ensemble.to_latex,
            loc="upper right",
            bbox=(0.925, 0.95),
        )
        log_dimension = np.log(self.ensemble.dimension)
        self.scale_limits_and_ticks(y=lambda value: value * log_dimension)

    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()
        self.create_time_figure(logarithmic_y=False)

        positive_time_mask = self.positive_time_mask
        positive_times = self.data.times[positive_time_mask]
        self.ax.plot(
            positive_times,
            self.data.entropy[positive_time_mask],
            color=self.entropy_color,
            alpha=self.entropy_alpha,
            linewidth=self.entropy_width,
            zorder=self.entropy_zorder,
        )
        self.ax.plot(
            positive_times,
            self.data.kl_divergence[positive_time_mask],
            color=self.kl_color,
            alpha=self.kl_alpha,
            linewidth=self.kl_width,
            zorder=self.kl_zorder,
        )
        self.finish_plot(path=path)
