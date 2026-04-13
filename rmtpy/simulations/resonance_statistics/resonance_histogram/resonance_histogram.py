from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..._histogram import Histogram
from ..._plot import PlotAxes, PlotLegend, Plot
from ....compounds import Compound
from ....ensembles import ManyBodyEnsemble, PoissonEnsemble, SachdevYeKitaevEnsemble
from ....utils import rmtpy_converter


@dataclass(repr=False, eq=False, kw_only=True)
class ResonanceHistogramLegend(PlotLegend):
    loc: str = "upper right"
    bbox: tuple[float, float] = (0.94, 0.95)


@dataclass(repr=False, eq=False, kw_only=True)
class ResonanceHistogramAxes(PlotAxes):
    xticks: tuple[float, ...] = (-1.0, 0.0, 1.0)  # units of energy_0
    xticks_minor: tuple[float, ...] = (-0.5, 0.5)
    xlabel: str = r"$\mathcal{E}$"
    xtick_labels: tuple[str, ...] = (
        r"$-E_0$",
        r"$0$",
        r"$E_0$",
    )

    yticks: tuple[float, ...] = (0.0, 1.0, 2.0)  # units of 1 / (pi * energy_0)
    yticks_minor: tuple[float, ...] = (0.0, 1.0, 2.0)
    ylabel: str = r"$\ensavg{\rho(\mathcal{E})}$"
    ytick_labels: tuple[str, ...] = (
        r"$0$",
        r"$\frac{1}{\pi E_0}$",
        r"$\frac{2}{\pi E_0}$",
    )

    poisson_yticks: tuple[float, ...] = (
        0.0,
        0.25 * np.pi,
        0.5 * np.pi,
        0.75 * np.pi,
    )  # units of 1 / (pi * energy_0)
    poisson_yticks_minor: tuple[float, ...] = (
        0.125 * np.pi,
        0.375 * np.pi,
        0.625 * np.pi,
        0.875 * np.pi,
    )
    poisson_ytick_labels: tuple[str, ...] = (
        r"$0$",
        r"$\frac{1}{4E_0}$",
        r"$\frac{1}{2E_0}$",
        r"$\frac{3}{4E_0}$",
    )

    syk2_yticks: tuple[float, ...] = tuple(range(6))  # units of 1 / (pi * energy_0)
    syk2_yticks_minor: tuple[float, ...] = tuple(x + 0.5 for x in range(6))
    syk2_ytick_labels: tuple[str, ...] = (
        r"$0$",
        r"$\frac{1}{\pi E_0}$",
        r"$\frac{2}{\pi E_0}$",
        r"$\frac{3}{\pi E_0}$",
        r"$\frac{4}{\pi E_0}$",
        r"$\frac{5}{\pi E_0}$",
    )

    syk4_yticks: tuple[float, ...] = tuple(range(3))  # units of 1 / (pi * energy_0)
    syk4_yticks_minor: tuple[float, ...] = tuple(x + 0.5 for x in range(3))
    syk4_ytick_labels: tuple[str, ...] = (
        r"$0$",
        r"$\frac{1}{\pi E_0}$",
        r"$\frac{2}{\pi E_0}$",
    )


@dataclass(repr=False, eq=False, kw_only=True)
class ResonanceHistogramPlot(Plot):
    data: Histogram
    axes: ResonanceHistogramAxes = field(default_factory=ResonanceHistogramAxes)
    num_points: int = 1000

    xlim: tuple[float, float] = (-1.2, 1.2)  # units of energy_0
    ylim: tuple[float, float] = (0.0, 2.6)  # units of 1 / (pi * energy_0)

    poisson_ylim: tuple[float, float] = (0.0, 1.25)  # units of 1 / (pi * energy_0)
    syk2_ylim: tuple[float, float] = (0.0, 4.0)
    syk4_ylim: tuple[float, float] = (0.0, 2.5)

    histogram_zorder: int = 1
    histogram_alpha: float = 0.5
    histogram_color: str = "OrangeRed"
    histogram_legend: str = "simulation"

    pdf_zorder: int = 2
    pdf_width: float = 2.0
    pdf_alpha: float = 1.0
    pdf_color: str = "Black"
    pdf_legend: str = "theory"

    legend_labels: tuple[str, str] = (histogram_legend, pdf_legend)
    legend_handles: tuple[Patch, Line2D] = (
        Patch(color=histogram_color, alpha=histogram_alpha),
        Line2D([0], [0], color=pdf_color, linewidth=pdf_width),
    )

    def set_derived_attributes(self) -> None:
        try:
            compound_meta: dict = self.data.metadata["simulation"]["args"]["compound"]
        except KeyError:
            raise ValueError("Compound metadata not found.")
        except TypeError:
            raise ValueError("Metadata is not properly structured.")

        self.compound: Compound = rmtpy_converter.structure(compound_meta, Compound)
        ensemble: ManyBodyEnsemble = self.compound.ensemble
        energy_0: float = ensemble.ground_state_energy

        self.legend: ResonanceHistogramLegend = ResonanceHistogramLegend(
            handles=self.legend_handles, labels=self.legend_labels
        )
        if self.legend.title is None:
            self.legend.title = self.compound.to_latex

        axes: ResonanceHistogramAxes = self.axes
        if type(ensemble) == PoissonEnsemble:
            self.ylim = self.poisson_ylim

            axes.ytick_labels = axes.poisson_ytick_labels
            axes.yticks = axes.poisson_yticks
            axes.yticks_minor = axes.poisson_yticks_minor

        elif type(ensemble) == SachdevYeKitaevEnsemble:
            if ensemble.q == 2:
                self.ylim = self.syk2_ylim

                axes.ytick_labels = axes.syk2_ytick_labels
                axes.yticks = axes.syk2_yticks
                axes.yticks_minor = axes.syk2_yticks_minor

            elif ensemble.q == 4:
                self.ylim = self.syk4_ylim

                axes.ytick_labels = axes.syk4_ytick_labels
                axes.yticks = axes.syk4_yticks
                axes.yticks_minor = axes.syk4_yticks_minor

        self.xlim = tuple(x * energy_0 for x in self.xlim)
        self.ylim = tuple(y / np.pi / energy_0 for y in self.ylim)

        axes.xticks = tuple(xtick * energy_0 for xtick in axes.xticks)
        axes.yticks = tuple(ytick / np.pi / energy_0 for ytick in axes.yticks)
        axes.xticks_minor = tuple(xtick * energy_0 for xtick in axes.xticks_minor)
        axes.yticks_minor = tuple(
            ytick / np.pi / energy_0 for ytick in axes.yticks_minor
        )

    def plot(self, path: str | Path) -> None:
        self.set_derived_attributes()

        self.create_figure()

        self.ax.hist(
            self.data.bins[:-1],
            bins=self.data.bins,
            weights=self.data.histogram,
            color=self.histogram_color,
            alpha=self.histogram_alpha,
            zorder=self.histogram_zorder,
        )

        energies: np.ndarray = np.linspace(self.xlim[0], self.xlim[1], self.num_points)
        resonance_pdf: np.ndarray = self.data.numerical_pdf(energies)

        self.ax.plot(
            energies,
            resonance_pdf,
            color=self.pdf_color,
            alpha=self.pdf_alpha,
            linewidth=self.pdf_width,
            zorder=self.pdf_zorder,
        )

        self.finish_plot(path=path)
