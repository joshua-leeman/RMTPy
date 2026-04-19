from __future__ import annotations

import inspect
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.axes import Axes
from numpy.lib.npyio import NpzFile

from ._data import DATA_REGISTRY, Data, _normalize_metadata, _normalize_source
from ..utils import rmtpy_converter


PLOT_REGISTRY: dict[str, type[Plot]] = {}


def plot_data(data_path: str | Path) -> None:
    data_path: Path = Path(data_path)
    out_dir: Path = data_path.parent

    plot: Plot = rmtpy_converter.structure(data_path, Plot)
    plot.plot(path=out_dir)


def _configure_matplotlib() -> None:
    rcParams["axes.axisbelow"] = False
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = "Latin Modern Roman"
    try:
        rcParams["text.usetex"] = True
        rcParams["text.latex.preamble"] = "\n".join(
            [
                r"\usepackage{amsmath}",
                r"\newcommand{\ensavg}[1]{\langle\hspace{-0.7ex}\langle #1 \hspace{-0.3ex} \rangle\hspace{-0.7ex}\rangle}",
                r"\newcommand{\diff}{\mathrm{d}}",
            ]
        )
    except:
        pass


@dataclass(repr=False, eq=False, kw_only=True)
class PlotAxes(ABC):
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


@dataclass(repr=False, eq=False, kw_only=True)
class PlotLegend(ABC):
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


@dataclass(repr=False, eq=False, kw_only=True)
class Plot(ABC):
    data: Data

    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None

    axes: PlotAxes = field(default_factory=PlotAxes)
    legend: PlotLegend = field(default_factory=PlotLegend)

    dpi: int = 300

    def __post_init__(self) -> None:
        _configure_matplotlib()

    def __init_subclass__(cls) -> None:
        if not inspect.isabstract(cls):
            plot_key: str = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", cls.__name__)
            plot_key = plot_key.lower()
            plot_key = plot_key.replace("_plot", "_data")
            PLOT_REGISTRY[plot_key] = cls

    @property
    def file_name(self) -> str:
        return self.data.file_name.replace("_data", "_plot")

    def create_figure(self) -> None:
        self.fig, self.ax = plt.subplots()
        plt.close(self.fig)

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


@rmtpy_converter.register_structure_hook
def plot_structure_hook(src: str | Path | dict[str, Any] | NpzFile | Plot, _) -> Plot:
    src_dict: dict[str, Any] = _normalize_source(src)
    metadata: dict[str, Any] = _normalize_metadata(src_dict["metadata"])
    src_dict["metadata"] = metadata

    plot_key: str | None = metadata.get("name", None)
    if plot_key in PLOT_REGISTRY:
        plot_cls: type[Plot] = PLOT_REGISTRY[plot_key]
    else:
        raise ValueError(f"No registered Plot class found in {src}")

    if plot_key in DATA_REGISTRY:
        data_cls: type[Data] = DATA_REGISTRY[plot_key]
    else:
        raise ValueError(f"No registered Data class found for Plot in {src}")

    data_inst: Data = rmtpy_converter.structure(src_dict, data_cls)
    return plot_cls(data=data_inst)
