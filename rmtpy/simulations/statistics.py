from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, TypeVar

import attrs
import numpy as np
from scipy.interpolate import PchipInterpolator

import rmtpy.density

from .data import Data
from .histogram import Histogram, finalize_histogram
from .histogram2D import Histogram2D, finalize_histogram2D
from .observable import Observable
from .plot import Plot

REALIZATIONS_METADATA: dict[str, str] = {
    "dir_name": "realizs",
    "latex_name": "R",
}
POLYNOMIAL_DEGREE_MIN: int = 2
POLYNOMIAL_DEGREE_STEP: int = 2

DataT = TypeVar("DataT", bound=Data)


def create_coefficient_histograms(
    *,
    prefix: str,
    max_degree: int,
    support: tuple[float, float],
    plot_cls: type[Plot] | None = None,
    unfolding: str = "raw",
) -> list[Observable]:
    return [
        create_histogram_observable(
            file_name=f"{prefix}_coeff_{degree}_histogram",
            support=support,
            plot_cls=plot_cls,
            metadata={
                "degree": degree,
                "unfolding": unfolding,
            },
        )
        for degree in range(1, max_degree + 1)
    ]


def create_degree_observables(
    *,
    degrees: Iterable[int],
    file_name_template: str,
    factory: Callable[[str, int], Observable],
) -> list[Observable]:
    return [
        factory(file_name_template.format(degree=degree), degree) for degree in degrees
    ]


def create_histogram_observable(
    *,
    file_name: str,
    support: tuple[float, float],
    log_base: float | None = None,
    num_bins: int | None = None,
    plot_cls: type[Plot] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Observable:
    histogram_kwargs: dict[str, Any] = {
        "file_name": file_name,
        "support": support,
    }
    if log_base is not None:
        histogram_kwargs["log_base"] = log_base
    if num_bins is not None:
        histogram_kwargs["num_bins"] = num_bins

    return create_observable(
        data=Histogram(**histogram_kwargs),
        plot_cls=plot_cls,
        metadata=metadata,
        finalize=finalize_histogram,
    )


def create_histogram2d_observable(
    *,
    file_name: str,
    x_support: tuple[float, float],
    y_support: tuple[float, float],
    x_log_base: float | None = None,
    y_log_base: float | None = None,
    x_num_bins: int | None = None,
    y_num_bins: int | None = None,
    plot_cls: type[Plot] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Observable:
    histogram_kwargs: dict[str, Any] = {
        "file_name": file_name,
        "x_support": x_support,
        "y_support": y_support,
    }

    if x_log_base is not None:
        histogram_kwargs["x_log_base"] = x_log_base
    if y_log_base is not None:
        histogram_kwargs["y_log_base"] = y_log_base
    if x_num_bins is not None:
        histogram_kwargs["x_num_bins"] = x_num_bins
    if y_num_bins is not None:
        histogram_kwargs["y_num_bins"] = y_num_bins

    return create_observable(
        data=Histogram2D(**histogram_kwargs),
        plot_cls=plot_cls,
        metadata=metadata,
        finalize=finalize_histogram2D,
    )


def create_observable(
    *,
    data: Data,
    plot_cls: type[Plot] | None = None,
    metadata: dict[str, Any] | None = None,
    finalize: Callable[[Data], None] | None = None,
) -> Observable:
    observable = Observable(
        data=data,
        plot_cls=plot_cls,
        finalize=finalize,
    )
    observable.metadata.update(metadata)
    return observable


def create_truncated_average_cdf_interpolators(
    density: rmtpy.density.DensityModel,
    degrees: Iterable[int],
    density_name: str,
) -> list[PchipInterpolator]:
    if density.average_coeffs is None:
        raise NotImplementedError(
            f"Truncated polynomial unfolding requires a polynomial {density_name} "
            "density polynomial expansion."
        )

    return [
        density.create_variate_cdf_interpolator(
            coeffs=truncate_coeffs(density.average_coeffs, degree)
        )
        for degree in degrees
    ]


def observable_data(observable: Observable, cls: type[DataT]) -> DataT:
    data: Data = observable.data
    if not isinstance(data, cls):
        raise TypeError(f"Expected {cls.__name__}, got {type(data).__name__}.")

    return data


def observable_data_list(
    observables: Iterable[Observable], cls: type[DataT]
) -> list[DataT]:
    return [observable_data(observable, cls) for observable in observables]


def nearest_neighbor_spacings(values: np.ndarray, degeneracy: int = 1) -> np.ndarray:
    spacings: np.ndarray = np.diff(np.sort(values))
    if degeneracy > 1:
        spacings = np.repeat(spacings[1::degeneracy], degeneracy)
    return spacings


def scale_support(support: tuple[float, float], scale: float) -> tuple[float, float]:
    return scale * support[0], scale * support[1]


def simulation_output_path(simulation: Any, root: Path) -> Path:
    path: Path = root
    for name, attr in attrs.fields_dict(type(simulation)).items():
        dir_name = attr.metadata.get("dir_name", None)
        if dir_name is None:
            continue

        value = re.sub(r"[^\w\-.]", "_", str(getattr(simulation, name)))
        path /= f"{dir_name}_{value.replace('.', 'p')}"
    return path


def truncated_polynomial_degrees(max_degree: int) -> range:
    return range(POLYNOMIAL_DEGREE_MIN, max_degree + 1, POLYNOMIAL_DEGREE_STEP)


def truncate_coeffs(coeffs: np.ndarray, truncate_degree: int) -> np.ndarray:
    truncated_coeffs: np.ndarray = np.zeros_like(coeffs)
    truncated_coeffs[: truncate_degree + 1] = coeffs[: truncate_degree + 1]
    return truncated_coeffs
