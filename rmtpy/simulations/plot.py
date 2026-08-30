from __future__ import annotations

import dataclasses
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from rmtpy.conversion import RMT_CONVERTER

from .data import Data
from .histogram import Histogram

DIMENSION_TIME_LOG_SUPPORT: tuple[float, float] = (-0.5, 1.5)

UNFOLDED_DIMENSION_TIME_LOG_SUPPORT: tuple[float, float] = (-1.5, 0.5)


def configure_matplotlib() -> None:
    matplotlib.rcParams["axes.axisbelow"] = False
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = "Latin Modern Roman"
    try:
        matplotlib.rcParams["text.usetex"] = True
        matplotlib.rcParams["text.latex.preamble"] = "\n".join(
            [
                r"\usepackage{amsmath}",
                (
                    r"\newcommand{\ensavg}[1]{"
                    r"\langle\hspace{-0.7ex}\langle #1 "
                    r"\rangle\hspace{-0.7ex}\rangle}"
                ),
                r"\newcommand{\diff}{\mathrm{d}}",
            ]
        )
    except (KeyError, ValueError) as exc:
        logging.getLogger(__name__).warning(
            "Could not configure LaTeX rendering for Matplotlib: %s", exc
        )


def plot_data(data_path: str | Path, *, plot_cls: type[Plot]) -> None:
    data_path = Path(data_path)
    plot_cls(data=Data.load(data_path)).plot(path=data_path.parent)


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class Plot(ABC):
    """Transient view constructed when an observable is written as a figure."""

    data: Data
    runtime_simulation_args: dict[str, Any] | None = dataclasses.field(
        default=None,
        repr=False,
    )

    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None

    axes: PlotAxes = dataclasses.field(default_factory=lambda: PlotAxes())
    legend: PlotLegend = dataclasses.field(default_factory=lambda: PlotLegend())

    dpi: int = 300

    def __post_init__(self) -> None:
        configure_matplotlib()

    @property
    def file_name(self) -> str:
        return self.data.file_name.removesuffix("_data") + "_plot"

    @property
    def simulation_args(self) -> dict[str, Any]:
        try:
            args = self.data.metadata["simulation"]["args"]
        except KeyError as exc:
            raise ValueError("Simulation metadata not found.") from exc
        except TypeError as exc:
            raise ValueError("Metadata is not properly structured.") from exc

        if not isinstance(args, dict):
            raise ValueError("Simulation args metadata is not properly structured.")
        return args

    def simulation_arg(self, key: str) -> Any:
        try:
            return self.simulation_args[key]
        except KeyError as exc:
            raise ValueError(f"Simulation arg metadata not found: {key}.") from exc

    def structure_simulation_arg(self, key: str, cls: type) -> Any:
        if self.runtime_simulation_args is not None:
            try:
                value = self.runtime_simulation_args[key]
            except KeyError as exc:
                raise ValueError(f"Simulation arg not found: {key}.") from exc

            if isinstance(value, cls):
                return value

            structured_value = RMT_CONVERTER.structure(value, cls)
            self.runtime_simulation_args[key] = structured_value
            return structured_value

        return RMT_CONVERTER.structure(self.simulation_arg(key), cls)

    def create_figure(self) -> None:
        self.fig, self.ax = plt.subplots()
        plt.close(self.fig)

    def draw_histogram(self, *, color: str, alpha: float, zorder: int) -> None:
        if not isinstance(self.data, Histogram):
            raise ValueError("Data must be a `Histogram` instance")

        self.ax.hist(
            self.data.bins[:-1],
            bins=self.data.bins,
            weights=self.data.histogram,
            color=color,
            alpha=alpha,
            zorder=zorder,
        )

    def scale_limits_and_ticks(
        self,
        *,
        x: Callable[[float], float] | None = None,
        y: Callable[[float], float] | None = None,
    ) -> None:
        if x is not None:
            if self.xlim is not None:
                self.xlim = tuple(x(value) for value in self.xlim)
            if self.axes.xticks is not None:
                self.axes.xticks = tuple(x(value) for value in self.axes.xticks)
            if self.axes.xticks_minor is not None:
                self.axes.xticks_minor = tuple(
                    x(value) for value in self.axes.xticks_minor
                )

        if y is not None:
            if self.ylim is not None:
                self.ylim = tuple(y(value) for value in self.ylim)
            if self.axes.yticks is not None:
                self.axes.yticks = tuple(y(value) for value in self.axes.yticks)
            if self.axes.yticks_minor is not None:
                self.axes.yticks_minor = tuple(
                    y(value) for value in self.axes.yticks_minor
                )

    def finish_plot(self, path: str | Path) -> None:
        if not hasattr(self, "fig") or not hasattr(self, "ax"):
            raise AttributeError(
                "Figure and axis not created. Call create_figure() first."
            )

        if self.xlim is not None:
            self.ax.set_xlim(self.xlim)
        if self.ylim is not None:
            self.ax.set_ylim(self.ylim)

        self.axes.configure(ax=self.ax)
        self.legend.configure(ax=self.ax)

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path / self.file_name, dpi=self.dpi, bbox_inches="tight")

    @abstractmethod
    def plot(self, path: str | Path) -> None:
        pass


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class PlotAxes:
    axes_width: float = 1.0

    xlabel: None = None
    xlabel_fontsize: int = 12
    ylabel: None = None
    ylabel_fontsize: int = 12

    xticks: tuple[float, ...] | None = None
    yticks: tuple[float, ...] | None = None
    xticks_minor: tuple[float, ...] | None = None
    yticks_minor: tuple[float, ...] | None = None

    tick_length: float = 6.0

    xtick_labels: tuple[str, ...] | None = None
    ytick_labels: tuple[str, ...] | None = None
    tick_fontsize: int = 10

    def configure(self, ax: Axes) -> None:
        for spine in ax.spines.values():
            spine.set_linewidth(self.axes_width)

        if self.xlabel is not None:
            ax.set_xlabel(self.xlabel, fontsize=self.xlabel_fontsize)
        if self.ylabel is not None:
            ax.set_ylabel(self.ylabel, fontsize=self.ylabel_fontsize)

        if self.xticks is not None:
            ax.set_xticks(self.xticks)
        if self.yticks is not None:
            ax.set_yticks(self.yticks)
        if self.xticks_minor is not None:
            ax.set_xticks(self.xticks_minor, minor=True)
        if self.yticks_minor is not None:
            ax.set_yticks(self.yticks_minor, minor=True)

        ax.tick_params(
            direction="in",
            top=True,
            bottom=True,
            left=True,
            right=True,
            which="both",
            length=self.tick_length,
        )

        if self.xtick_labels is not None:
            ax.set_xticklabels(self.xtick_labels, fontsize=self.tick_fontsize)
        else:
            ax.tick_params(axis="x", labelsize=self.tick_fontsize)

        if self.ytick_labels is not None:
            ax.set_yticklabels(self.ytick_labels, fontsize=self.tick_fontsize)
        else:
            ax.tick_params(axis="y", labelsize=self.tick_fontsize)


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class DimensionTimeAxes(PlotAxes):
    """Shared raw-time axis for dimension-scaled dynamical observables."""

    xticks: tuple[float, ...] = (0.0, 0.5, 1.0)  # log scale base dimension
    # t_0 = j_\text{\tiny 1,1} / J
    xlabel: str = r"$u = t / t_0$"
    xtick_labels: tuple[str, ...] = (
        r"$N_\text{m}^{-1}$",
        r"$D^{1/2} N_\text{m}^{-1}$",
        r"$D N_\text{m}^{-1}$",
    )


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class UnfoldedDimensionTimeAxes(PlotAxes):
    """Shared unfolded-time axis in units of the Heisenberg time."""

    xticks: tuple[float, ...] = (-1.0, -0.5, 0.0)  # log scale base dimension
    xlabel: str = r"$\upsilon = \tau / \tau_\text{\tiny H}$"
    xtick_labels: tuple[str, ...] = (
        r"$D^{-1}$",
        r"$D^{-1/2}$",
        r"$1$",
    )


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class PlotLegend:
    handles: tuple | None = None
    labels: tuple[str, ...] | None = None

    fontsize: int = 10
    textalignment: str = "left"

    on_black_background: bool = False

    title: str | None = None
    title_fontsize: int = 10
    title_linespacing: float = 1.5

    loc: str = "best"
    bbox: tuple[float, float] | None = None
    frameon: bool = False

    def configure(self, ax: Axes) -> None:
        if self.handles is not None and self.labels is not None:
            legend = ax.legend(
                handles=self.handles,
                labels=self.labels,
                title=self.title,
                loc=self.loc,
                bbox_to_anchor=self.bbox,
                frameon=self.frameon,
                fontsize=self.fontsize,
                title_fontsize=self.title_fontsize,
                alignment=self.textalignment,
            )

            legend.get_title().set_linespacing(self.title_linespacing)

            if self.on_black_background:
                legend.get_title().set_color("white")
                for text in legend.get_texts():
                    text.set_color("white")
