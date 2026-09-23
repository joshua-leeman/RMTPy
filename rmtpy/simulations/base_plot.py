import dataclasses
import logging
import os
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal, Protocol, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend

from ..conversion import RMT_CONVERTER
from .base_data import Data
from .base_simulation import RunContext
from .histogram import Histogram

type LegendAlignment = Literal["left", "center", "right"]
type LegendLocation = Literal[
    "best",
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "right",
    "center left",
    "center right",
    "lower center",
    "upper center",
    "center",
]


class Spine(Protocol):
    def set_linewidth(self, width: float) -> object: ...


class ConfigurableAxes(Protocol):
    spines: Mapping[str, Spine]

    def set_xlabel(self, xlabel: str, *, fontsize: float = ...) -> object: ...

    def set_ylabel(self, ylabel: str, *, fontsize: float = ...) -> object: ...

    def set_xticks(self, ticks: Sequence[float], *, minor: bool = False) -> object: ...

    def set_yticks(self, ticks: Sequence[float], *, minor: bool = False) -> object: ...

    def set_xticklabels(
        self, labels: Sequence[str], *, fontsize: float = ...
    ) -> object: ...

    def set_yticklabels(
        self, labels: Sequence[str], *, fontsize: float = ...
    ) -> object: ...

    def tick_params(self, *args: object, **kwargs: object) -> object: ...


def _configure_matplotlib() -> None:
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


def _axis_limits(limits: tuple[float, ...], /) -> tuple[float, float]:
    if len(limits) != 2:
        raise ValueError("`xlim` and `ylim` must contain exactly two values.")

    return (limits[0], limits[1])


def _plot_configuration(value: object) -> object:
    if not isinstance(value, dict):
        return value

    mapping = cast(dict[object, object], value)
    parameters_value = mapping.get("parameters")
    if set(mapping) == {"type", "parameters"} and isinstance(parameters_value, dict):
        parameters = cast(dict[object, object], parameters_value)
        if "seed" in parameters:
            parameters["seed"] = 0

        for key, item in parameters.items():
            parameters[key] = _plot_configuration(item)

        mapping["parameters"] = parameters
        return mapping

    return {key: _plot_configuration(item) for key, item in mapping.items()}


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class PlotAxes:
    axes_width: float = 1.0

    xlabel: str = ""
    ylabel: str = ""
    ylabel_fontsize: int = 12
    xlabel_fontsize: int = 12

    xticks: tuple[float, ...] = ()
    yticks: tuple[float, ...] = ()
    xticks_minor: tuple[float, ...] = ()
    yticks_minor: tuple[float, ...] = ()

    tick_length: float = 6.0
    tick_fontsize: int = 10

    xtick_labels: tuple[str, ...] = ()
    ytick_labels: tuple[str, ...] = ()

    def configure(self, axes: ConfigurableAxes) -> None:
        for spine in axes.spines.values():
            _ = spine.set_linewidth(self.axes_width)

        if self.xlabel:
            _ = axes.set_xlabel(self.xlabel, fontsize=self.xlabel_fontsize)
        if self.ylabel:
            _ = axes.set_ylabel(self.ylabel, fontsize=self.ylabel_fontsize)

        if self.xticks:
            _ = axes.set_xticks(self.xticks)
        if self.yticks:
            _ = axes.set_yticks(self.yticks)

        if self.xticks_minor:
            _ = axes.set_xticks(self.xticks_minor, minor=True)
        if self.yticks_minor:
            _ = axes.set_yticks(self.yticks_minor, minor=True)

        _ = axes.tick_params(
            direction="in",
            top=True,
            bottom=True,
            left=True,
            right=True,
            which="both",
            length=self.tick_length,
        )

        if self.xtick_labels:
            _ = axes.set_xticklabels(self.xtick_labels, fontsize=self.tick_fontsize)
        else:
            _ = axes.tick_params(axis="x", labelsize=self.tick_fontsize)

        if self.ytick_labels:
            _ = axes.set_yticklabels(self.ytick_labels, fontsize=self.tick_fontsize)
        else:
            _ = axes.tick_params(axis="y", labelsize=self.tick_fontsize)


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class LogDimensionTimeAxes(PlotAxes):
    xticks: tuple[float, ...] = (0.0, 0.5, 1.0)  # log scale base dimension

    xlabel: str = r"$u = t / t_0$"  # t_0 = j_\text{\tiny 1,1} / J

    xtick_labels: tuple[str, ...] = (
        r"$N_\text{m}^{-1}$",
        r"$D^{1/2} N_\text{m}^{-1}$",
        r"$D N_\text{m}^{-1}$",
    )


DimensionTimeAxes = LogDimensionTimeAxes


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class LogDimensionUnfoldedTimeAxes(PlotAxes):
    xticks: tuple[float, ...] = (-1.0, -0.5, 0.0)  # log scale base dimension

    xlabel: str = r"$\upsilon = \tau / \tau_\text{\tiny H}$"

    xtick_labels: tuple[str, ...] = (
        r"$D^{-1}$",
        r"$D^{-1/2}$",
        r"$1$",
    )


UnfoldedDimensionTimeAxes = LogDimensionUnfoldedTimeAxes


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class PlotLegend:
    handles: tuple[Artist | tuple[Artist, ...], ...] = ()
    labels: tuple[str, ...] = ()

    fontsize: int = 10
    textalignment: LegendAlignment = "left"

    on_black_background: bool = False

    title: str = ""
    title_fontsize: int = 10
    title_linespacing: float = 1.5

    loc: LegendLocation = "best"
    bbox: tuple[float, ...] = ()
    frameon: bool = False

    def configure(self, ax: Axes) -> None:
        if self.handles and self.labels:
            configure_legend = cast(Callable[..., Legend], ax.legend)
            legend = configure_legend(
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


@dataclasses.dataclass(kw_only=True, eq=False)
class Plot(ABC):
    data: Data

    context: RunContext = dataclasses.field(repr=False)

    fig: Figure = dataclasses.field(init=False, repr=False)
    ax: Axes = dataclasses.field(init=False, repr=False)

    xlim: tuple[float, ...] = ()
    ylim: tuple[float, ...] = ()

    axes: PlotAxes = dataclasses.field(default_factory=PlotAxes)
    legend: PlotLegend = dataclasses.field(default_factory=PlotLegend)

    dpi: int = 300

    _structured_args: dict[tuple[str, str], object] = dataclasses.field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        _configure_matplotlib()

    @property
    def file_name(self) -> str:
        return self.data.file_name.removesuffix("_data") + "_plot"

    @abstractmethod
    def plot(self, path: str | Path) -> None:
        pass

    def calibration_coefficients(self, density: str) -> np.ndarray | None:
        calibration = self.context.execution.get("calibration")
        if not calibration:
            return None

        if not isinstance(calibration, Mapping):
            raise ValueError("Run calibration information is malformed.")

        calibration_mapping = cast(Mapping[str, object], calibration)
        if calibration_mapping.get("density") != density:
            return None

        coefficients = calibration_mapping.get("average_coefficients")
        if coefficients is None:
            raise ValueError("Run calibration coefficients are missing.")

        values = np.asarray(coefficients, dtype=np.float64)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError("Run calibration coefficients are malformed.")

        return values

    def build_figure(self) -> None:
        self.fig, self.ax = plt.subplots()

    def draw_histogram(self, *, color: str, alpha: float, zorder: int) -> None:
        if not isinstance(self.data, Histogram):
            raise ValueError("Data must be a `Histogram` instance")

        histogram = cast(Callable[..., object], self.ax.hist)
        _ = histogram(
            self.data.bins[:-1],
            bins=self.data.bins.tolist(),
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
            if self.xlim:
                self.xlim = (x(self.xlim[0]), x(self.xlim[1]))
            self.axes.xticks = tuple(x(value) for value in self.axes.xticks)
            self.axes.xticks_minor = tuple(x(value) for value in self.axes.xticks_minor)

        if y is not None:
            if self.ylim:
                self.ylim = (y(self.ylim[0]), y(self.ylim[1]))
            self.axes.yticks = tuple(y(value) for value in self.axes.yticks)
            self.axes.yticks_minor = tuple(y(value) for value in self.axes.yticks_minor)

    def finish_plot(self, path: str | Path) -> None:
        if not hasattr(self, "fig") or not hasattr(self, "ax"):
            raise AttributeError(
                "Figure and axis not yet created. Call build_figure() first."
            )

        try:
            if self.xlim:
                _ = self.ax.set_xlim(_axis_limits(self.xlim))
            if self.ylim:
                _ = self.ax.set_ylim(_axis_limits(self.ylim))

            self.axes.configure(axes=cast(ConfigurableAxes, cast(object, self.ax)))
            self.legend.configure(ax=self.ax)

            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            destination = path / f"{self.file_name}.png"
            if destination.exists():
                raise FileExistsError(f"Plot already exists: {destination}")

            temporary = path / f".{self.file_name}.{uuid.uuid4().hex}.partial.png"
            try:
                save_figure = cast(Callable[..., None], self.fig.savefig)
                save_figure(
                    temporary,
                    format="png",
                    dpi=self.dpi,
                    bbox_inches="tight",
                )
                with open(temporary, "rb") as file:
                    os.fsync(file.fileno())

                os.link(temporary, destination)
                descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))

                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            finally:
                temporary.unlink(missing_ok=True)
        finally:
            figure = cast(object, self.fig)
            if isinstance(figure, Figure):
                plt.close(figure)

    def structure_simulation_arg[T](self, key: str, cls: type[T]) -> T:
        cache_key = (key, f"{cls.__module__}.{cls.__qualname__}")
        if cache_key in self._structured_args:
            return cast(T, self._structured_args[cache_key])

        parameters = self.context.simulation_config["parameters"]
        if not isinstance(parameters, dict):
            raise ValueError("Simulation configuration parameters are malformed.")

        try:
            value = parameters[key]
        except KeyError as exc:
            raise ValueError(f"Simulation parameter not found: {key}.") from exc

        if isinstance(value, cls):
            value = cast(object, RMT_CONVERTER.unstructure(value))

        structured_value = RMT_CONVERTER.structure(_plot_configuration(value), cls)
        self._structured_args[cache_key] = structured_value
        return structured_value
