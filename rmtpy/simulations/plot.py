from __future__ import annotations

import dataclasses
import inspect
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from numpy.lib.npyio import NpzFile

from rmtpy.conversion import RMT_CONVERTER

from .data import REGISTRY as DATA_REGISTRY
from .data import Data, normalize_metadata, normalize_source

PLOT_REGISTRY: dict[str, type[Plot]] = {}


def plot_data(data_path: str | Path) -> None:
    data_path: Path = Path(data_path)
    out_dir: Path = data_path.parent

    plot: Plot = RMT_CONVERTER.structure(data_path, Plot)
    plot.plot(path=out_dir)


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
                    r"\hspace{-0.3ex} \rangle\hspace{-0.7ex}\rangle}"
                ),
                r"\newcommand{\diff}{\mathrm{d}}",
            ]
        )
    except (KeyError, ValueError) as exc:
        logging.getLogger(__name__).warning(
            "Could not configure LaTeX rendering for Matplotlib: %s", exc
        )


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class PlotAxes:
    axes_width: float = 1.0

    xlabel: str = r"$x$"
    xlabel_fontsize: int = 12
    ylabel: str = r"$y$"
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

        ax.set_xlabel(self.xlabel, fontsize=self.xlabel_fontsize)
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
class PlotLegend:
    handles: tuple | None = None
    labels: tuple[str, ...] | None = None
    fontsize: int = 10
    textalignment: str = "left"

    title: str | None = None
    title_fontsize: int = 10

    loc: str = "best"
    bbox: tuple[float, float] | None = None
    frameon: bool = False

    def configure(self, ax: Axes) -> None:
        if self.handles is not None and self.labels is not None:
            ax.legend(
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


@dataclasses.dataclass(repr=False, eq=False, kw_only=True)
class Plot(ABC):
    data: Data

    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None

    axes: PlotAxes = dataclasses.field(default_factory=PlotAxes)
    legend: PlotLegend = dataclasses.field(default_factory=PlotLegend)

    dpi: int = 300

    def __post_init__(self) -> None:
        configure_matplotlib()

    def __init_subclass__(cls) -> None:
        if not inspect.isabstract(cls):
            plot_key: str = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", cls.__name__)
            plot_key = plot_key.lower()
            plot_key = plot_key.replace("_plot", "_data")
            PLOT_REGISTRY[plot_key] = cls

    @property
    def file_name(self) -> str:
        return self.data.file_name.replace("_data", "_plot")

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
        return RMT_CONVERTER.structure(self.simulation_arg(key), cls)

    def create_figure(self) -> None:
        self.fig, self.ax = plt.subplots()
        plt.close(self.fig)

    def draw_histogram(self, *, color: str, alpha: float, zorder: int) -> None:
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

        path: Path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path / self.file_name, dpi=self.dpi, bbox_inches="tight")

    @abstractmethod
    def plot(self, path: str | Path) -> None:
        pass


@RMT_CONVERTER.register_structure_hook
def plot_structure_hook(src: str | Path | dict[str, Any] | NpzFile | Plot, _) -> Plot:
    src_dict: dict[str, Any] = normalize_source(src)
    metadata: dict[str, Any] = normalize_metadata(src_dict["metadata"])
    src_dict["metadata"] = metadata

    plot_key: str | None = metadata.get("name")
    if plot_key in PLOT_REGISTRY:
        plot_cls: type[Plot] = PLOT_REGISTRY[plot_key]
    else:
        raise ValueError(f"No registered Plot class found in {src}")

    if plot_key in DATA_REGISTRY:
        data_cls: type[Data] = DATA_REGISTRY[plot_key]
    else:
        raise ValueError(f"No registered Data class found for Plot in {src}")

    data_inst: Data = RMT_CONVERTER.structure(src_dict, data_cls)
    return plot_cls(data=data_inst)
