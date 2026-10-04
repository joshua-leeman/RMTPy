import dataclasses
import logging
import math
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, MaxNLocator

from ..compounds import CompoundEnsemble
from ..conversion import RMT_CONVERTER, unwrap_json_value
from ..universal import wigner_surmise
from .base_data import Data
from .base_simulation import SimulationManifest
from .histograms import Histogram

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
type SpacingReference = tuple[
    str,
    int,
    str,
    Literal["solid", "dashed", "dashdot", "dotted"],
]

UNFOLDING_LABELS_BY_TYPE: dict[str, str] = {
    "average": "Ave",
    "variate": "Var",
    "weight": "Wgt",
}

ENSEMBLE_AVERAGED_CURVE_WIDTH: float = 1.4
SINGLE_REALIZATION_CURVE_WIDTH: float = 0.5
CURVE_WIDTH: float = ENSEMBLE_AVERAGED_CURVE_WIDTH

FORM_FACTOR_COLOR: str = "#0072B2"
CONNECTED_FORM_FACTOR_COLOR: str = "#D54300"
SINGLE_REALIZATION_FORM_FACTOR_COLOR: str = "#009E73"

SPACING_REFERENCES: tuple[SpacingReference, ...] = (
    ("GOE", 1, "#0072B2", "solid"),
    ("GUE", 2, "#009E73", "dashed"),
    ("GSE", 4, "#CC79A7", "dashdot"),
    ("Poisson", 0, "#000000", "dotted"),
)
SPACING_REFERENCE_LABELS: tuple[str, ...] = tuple(
    label for label, _, _, _ in SPACING_REFERENCES
)

_MAJOR_TICK_STEPS: tuple[float, ...] = (1.0, 2.0, 2.5, 5.0, 10.0)


class Spine(Protocol):
    def set_linewidth(self, width: float) -> object: ...


class ConfigurableAxes(Protocol):
    @property
    def spines(self) -> Mapping[str, Spine]: ...

    def get_xscale(self) -> str: ...

    def get_yscale(self) -> str: ...

    def set_title(self, title: str, *, fontsize: float = ...) -> object: ...

    def set_xlabel(self, xlabel: str, *, fontsize: float = ...) -> object: ...

    def set_ylabel(
        self,
        ylabel: str,
        *,
        fontsize: float = ...,
        rotation: float = ...,
        labelpad: float = ...,
    ) -> object: ...

    def set_xticks(self, ticks: Sequence[float], *, minor: bool = False) -> object: ...

    def set_yticks(self, ticks: Sequence[float], *, minor: bool = False) -> object: ...

    def set_xticklabels(
        self, labels: Sequence[str], *, fontsize: float = ...
    ) -> object: ...

    def set_yticklabels(
        self, labels: Sequence[str], *, fontsize: float = ...
    ) -> object: ...

    def tick_params(
        self,
        axis: Literal["x", "y", "both"] = "both",
        *,
        which: Literal["major", "minor", "both"] = "major",
        direction: Literal["in", "out", "inout"] = "in",
        top: bool = True,
        bottom: bool = True,
        left: bool = True,
        right: bool = True,
        labelbottom: bool = True,
        labeltop: bool = True,
        labelleft: bool = True,
        labelright: bool = True,
        length: float = ...,
        labelsize: float = ...,
    ) -> object: ...


def nice_major_ticks(
    lower: float,
    upper: float,
    *,
    num_intervals: int = 5,
    symmetric: bool = False,
) -> tuple[float, ...]:
    locator = MaxNLocator(
        nbins=num_intervals,
        steps=_MAJOR_TICK_STEPS,
        min_n_ticks=3,
        symmetric=symmetric,
    )
    return tuple(float(value) for value in locator.tick_values(lower, upper))


def major_tick_step(major_ticks: tuple[float, ...]) -> float:
    return min(
        right - left
        for left, right in zip(major_ticks, major_ticks[1:], strict=False)
        if right > left
    )


def minor_ticks(major_ticks: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(
        0.5 * (major_ticks[index] + major_ticks[index + 1])
        for index in range(len(major_ticks) - 1)
    )


def latex_tick_labels(
    major_ticks: tuple[float, ...],
    *,
    show_positive_sign: bool,
    min_decimal_places: int = 0,
) -> tuple[str, ...]:
    decimal_places = 0
    if len(major_ticks) > 1:
        step = major_tick_step(major_ticks)
        decimal_places = max(0, -math.floor(math.log10(step)))
        scaled_step = step * math.pow(10.0, decimal_places)
        if not math.isclose(
            scaled_step,
            round(scaled_step),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            decimal_places += 1
    decimal_places = max(decimal_places, min_decimal_places)

    step = major_ticks[1] - major_ticks[0] if len(major_ticks) > 1 else 1.0
    zero_tolerance = abs(step) * 1e-9

    labels: list[str] = []
    for tick in major_ticks:
        value = 0.0 if abs(tick) <= zero_tolerance else tick
        sign = "+" if show_positive_sign and value > 0.0 else ""
        labels.append(rf"${sign}{value:.{decimal_places}f}$")

    return tuple(labels)


def configure_coefficient_histogram_axes(
    data: Histogram,
    axes: PlotAxes,
    /,
) -> tuple[tuple[float, float], tuple[float, float]]:
    total_count = int(np.sum(data.counts))
    if total_count:
        cumulative_counts = np.cumsum(data.counts)
        quantile_indices = np.searchsorted(
            cumulative_counts,
            np.multiply((0.01, 0.99), total_count),
            side="left",
        )
        lower_index, upper_index = cast(list[int], quantile_indices.tolist())
        lower = float(cast(np.floating, data.bins[lower_index]))
        upper = float(cast(np.floating, data.bins[upper_index + 1]))
    else:
        lower = float(cast(np.floating, data.bins[0]))
        upper = float(cast(np.floating, data.bins[-1]))

    occupied_extent = max(abs(lower), abs(upper))
    enclosing_ticks = nice_major_ticks(
        -occupied_extent,
        occupied_extent,
        num_intervals=4,
        symmetric=True,
    )
    axes.xticks = enclosing_ticks
    horizontal_padding = 0.5 * major_tick_step(axes.xticks)
    xlim = (
        axes.xticks[0] - horizontal_padding,
        axes.xticks[-1] + horizontal_padding,
    )
    axes.xticks_minor = minor_ticks(axes.xticks)
    axes.xtick_labels = latex_tick_labels(
        axes.xticks,
        show_positive_sign=True,
    )

    histogram_peak = float(np.max(data.histogram, initial=0.0))
    padded_peak = histogram_peak * (1.0 + 0.05)
    y_tick_target = padded_peak if padded_peak > 0.0 else 1.0
    axes.yticks = nice_major_ticks(
        0.0,
        y_tick_target,
    )
    ylim = (0.0, axes.yticks[-1])
    axes.yticks_minor = minor_ticks(axes.yticks)
    axes.ytick_labels = latex_tick_labels(
        axes.yticks,
        show_positive_sign=False,
        min_decimal_places=1,
    )

    return xlim, ylim


def format_coupling_label(compound: CompoundEnsemble, /) -> str:
    mean_coupling_squared = float(np.mean(compound.couplings**2))
    coupling_exponent = math.log10(
        mean_coupling_squared / compound.ensemble.spectral_radius
    )
    if abs(coupling_exponent) < 0.005:
        coupling_exponent = 0.0

    return rf"$\alpha = {{{coupling_exponent:.1f}}}$"


def configure_form_factor_axes(
    axes: Axes,
    /,
    *,
    dimension: int,
    x_tick_count: int,
    y_tick_count: int,
) -> None:
    set_xscale = cast(Callable[..., object], axes.set_xscale)
    set_yscale = cast(Callable[..., object], axes.set_yscale)
    _ = set_xscale("log", base=dimension)
    _ = set_yscale("log", base=dimension)

    set_x_locator = cast(Callable[..., object], axes.xaxis.set_major_locator)
    set_y_locator = cast(Callable[..., object], axes.yaxis.set_major_locator)
    _ = set_x_locator(LogLocator(base=dimension, numticks=x_tick_count))
    _ = set_y_locator(LogLocator(base=dimension, numticks=y_tick_count))


def configure_form_factor_minor_ticks(axes: ConfigurableAxes, /) -> None:
    _ = axes.tick_params(
        axis="both",
        which="minor",
        bottom=False,
        top=False,
        left=False,
        right=False,
    )


def spacing_legend_handles(
    *,
    histogram_color: str,
    histogram_alpha: float,
    width: float,
    alpha: float,
) -> tuple[Patch | Line2D, ...]:
    return (
        Patch(color=histogram_color, alpha=histogram_alpha),
        *(
            Line2D(
                [0],
                [0],
                color=color,
                linestyle=style,
                linewidth=width,
                alpha=alpha,
                label=label,
            )
            for label, _, color, style in SPACING_REFERENCES
        ),
    )


def draw_spacing_references(
    plot: Plot,
    spacings: np.ndarray[tuple[int], np.dtype[np.floating]],
    /,
    *,
    width: float,
    alpha: float,
    zorder: int,
    mean_spacing: float = 1.0,
) -> None:
    for label, dyson_index, color, style in SPACING_REFERENCES:
        values = (
            wigner_surmise(spacings / mean_spacing, dyson_index=dyson_index)
            / mean_spacing
        )
        plot.draw_curve(
            spacings,
            values,
            color=color,
            style=style,
            width=width,
            alpha=alpha,
            zorder=zorder,
            label=label,
        )


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
    else:
        value = cast(dict[object, object], value)

    mapping = {key: _plot_configuration(item) for key, item in value.items()}
    parameters = mapping.get("parameters")
    if (
        set(mapping) == {"type", "parameters"}
        and isinstance(parameters, dict)
        and "seed" in parameters
    ):
        parameters["seed"] = 0

    return mapping


@dataclasses.dataclass(kw_only=True, eq=False, weakref_slot=False)
class PlotAxes:
    axes_width: float = 1.0

    title: str = ""
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

        is_log_log = axes.get_xscale() == axes.get_yscale() == "log"

        if self.title:
            _ = axes.set_title(self.title, fontsize=self.xlabel_fontsize)
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
        elif is_log_log and len(self.xticks) > 1:
            _ = axes.set_xticks(
                tuple(
                    math.sqrt(left_tick * right_tick)
                    for left_tick, right_tick in zip(
                        self.xticks,
                        self.xticks[1:],
                        strict=False,
                    )
                ),
                minor=True,
            )
        if self.yticks_minor:
            _ = axes.set_yticks(self.yticks_minor, minor=True)
        elif is_log_log and len(self.yticks) > 1:
            _ = axes.set_yticks(
                tuple(
                    math.sqrt(left_tick * right_tick)
                    for left_tick, right_tick in zip(
                        self.yticks,
                        self.yticks[1:],
                        strict=False,
                    )
                ),
                minor=True,
            )

        _ = axes.tick_params(
            direction="in",
            top=True,
            bottom=True,
            left=True,
            right=True,
            which="both",
            length=self.tick_length,
        )

        if is_log_log:
            _ = axes.tick_params(
                axis="both",
                which="minor",
                length=self.tick_length / 2,
                labelbottom=False,
                labeltop=False,
                labelleft=False,
                labelright=False,
            )

        if self.xtick_labels:
            _ = axes.set_xticklabels(self.xtick_labels, fontsize=self.tick_fontsize)
        else:
            _ = axes.tick_params(axis="x", labelsize=self.tick_fontsize)

        if self.ytick_labels:
            _ = axes.set_yticklabels(self.ytick_labels, fontsize=self.tick_fontsize)
        else:
            _ = axes.tick_params(axis="y", labelsize=self.tick_fontsize)


@dataclasses.dataclass(kw_only=True, eq=False, weakref_slot=False)
class LogDimensionTimeAxes(PlotAxes):
    xticks: tuple[float, ...] = (0.0, 0.5, 1.0)  # log scale base dimension

    xlabel: str = r"$u = t / t_0$"  # t_0 = j_\text{\tiny 1,1} / J

    xtick_labels: tuple[str, ...] = (
        r"$N_\text{m}^{-1}$",
        r"$D^{1/2} N_\text{m}^{-1}$",
        r"$D N_\text{m}^{-1}$",
    )


@dataclasses.dataclass(kw_only=True, eq=False, weakref_slot=False)
class LogDimensionUnfoldedTimeAxes(PlotAxes):
    xticks: tuple[float, ...] = (-1.0, -0.5, 0.0)  # log scale base dimension

    xlabel: str = r"$\upsilon = \tau / \tau_\text{\tiny H}$"

    xtick_labels: tuple[str, ...] = (
        r"$D^{-1}$",
        r"$D^{-1/2}$",
        r"$1$",
    )


@dataclasses.dataclass(kw_only=True, eq=False, weakref_slot=False)
class PlotLegend:
    handles: tuple[Artist | tuple[Artist, ...], ...] = ()
    labels: tuple[str, ...] = ()

    fontsize: int = 10
    textalignment: LegendAlignment = "left"

    on_black_background: bool = False

    loc: LegendLocation = "best"
    bbox: tuple[float, ...] = ()
    frameon: bool = False

    def configure(self, ax: Axes) -> None:
        if self.handles and self.labels:
            configure_legend = cast(Callable[..., Legend], ax.legend)
            legend = configure_legend(
                handles=self.handles,
                labels=self.labels,
                loc=self.loc,
                bbox_to_anchor=self.bbox,
                frameon=self.frameon,
                fontsize=self.fontsize,
                alignment=self.textalignment,
            )

            if self.on_black_background:
                for text in legend.get_texts():
                    text.set_color("white")


@dataclasses.dataclass(kw_only=True, eq=False)
class Plot(ABC):
    data: Data

    context: SimulationManifest = dataclasses.field(repr=False)

    fig: Figure = dataclasses.field(init=False, repr=False)
    ax: Axes = dataclasses.field(init=False, repr=False)

    xlim: tuple[float, ...] = ()
    ylim: tuple[float, ...] = ()

    axes: PlotAxes = dataclasses.field(default_factory=PlotAxes)
    legend: PlotLegend = dataclasses.field(default_factory=PlotLegend)

    dpi: int = 300

    _derived_attributes_are_set: bool = dataclasses.field(
        default=False,
        init=False,
        repr=False,
    )

    _structured_args: dict[tuple[str, str], object] = dataclasses.field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        _configure_matplotlib()

    @property
    def file_name(self) -> str:
        return self.data._file_name.removesuffix("_data") + "_plot"

    @abstractmethod
    def plot(self, path: str | Path) -> None:
        pass

    def calibration_coefficients(self, density: str) -> np.ndarray | None:
        calibration = unwrap_json_value(self.context.execution.get("calibration"))
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

    def draw_curve(
        self,
        coordinates: np.ndarray[tuple[int], np.dtype[np.floating]],
        values: np.ndarray[tuple[int], np.dtype[np.floating]],
        /,
        *,
        color: str,
        alpha: float,
        width: float,
        zorder: int,
        label: str,
        style: str = "solid",
    ) -> None:
        plot = cast(Callable[..., object], self.ax.plot)
        _ = plot(
            coordinates,
            values,
            color=color,
            alpha=alpha,
            linewidth=width,
            zorder=zorder,
            label=label,
            linestyle=style,
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
            raise AttributeError("Figure and axis not yet created.")

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

    def store_manifest_arg[T](self, key: str, cls: type[T]) -> T:
        cache_key = (key, f"{cls.__module__}.{cls.__qualname__}")
        if cache_key in self._structured_args:
            return cast(T, self._structured_args[cache_key])

        parameters = self.context.configuration["parameters"]
        if not isinstance(parameters, dict):
            raise ValueError("Simulation configuration parameters are malformed.")

        try:
            value = parameters[key]
        except KeyError as exc:
            raise ValueError(f"Simulation parameter not found: {key}.") from exc

        value = unwrap_json_value(value)
        if isinstance(value, cls):
            value = cast(object, RMT_CONVERTER.unstructure(value))

        structured_value = RMT_CONVERTER.structure(_plot_configuration(value), cls)
        self._structured_args[cache_key] = structured_value
        return structured_value
