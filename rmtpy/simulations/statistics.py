from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np

from .data import Data
from .histogram import Histogram, finalize_histogram
from .histogram2D import Histogram2D, finalize_histogram2D
from .observable import Observable
from .plot import Plot

DataT = TypeVar("DataT", bound=Data)
Support = tuple[float, float]

POLYNOMIAL_DEGREE_MIN: int = 2
POLYNOMIAL_DEGREE_STEP: int = 2

REALIZATIONS_METADATA: dict[str, str] = {
    "dir_name": "realizs",
    "latex_name": "R",
}


def create_coefficient_histograms(
    *,
    prefix: str,
    max_degree: int,
    support: Support,
    plot_cls: type[Plot] | None = None,
    unfolding: str = "raw",
) -> list[Observable[Histogram]]:
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


def create_histogram2d_observable(
    *,
    file_name: str,
    x_support: Support,
    y_support: Support,
    x_log_base: float | None = None,
    y_log_base: float | None = None,
    x_num_bins: int | None = None,
    y_num_bins: int | None = None,
    plot_cls: type[Plot] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Observable[Histogram2D]:
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


def create_histogram_observable(
    *,
    file_name: str,
    support: Support,
    log_base: float | None = None,
    num_bins: int | None = None,
    plot_cls: type[Plot] | None = None,
    metadata: dict[str, Any] | None = None,
    finalize: Callable[[Histogram], None] | None = finalize_histogram,
) -> Observable[Histogram]:
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
        finalize=finalize,
    )


def create_observable(
    *,
    data: DataT,
    plot_cls: type[Plot] | None = None,
    metadata: dict[str, Any] | None = None,
    finalize: Callable[[Data], None] | None = None,
) -> Observable[DataT]:
    observable = Observable(
        data=data,
        plot_cls=plot_cls,
        finalize=finalize,
    )
    if metadata is not None:
        observable.metadata.update(metadata)

    return observable


def nearest_neighbor_spacings(values: np.ndarray, *, degeneracy: int = 1) -> np.ndarray:
    spacings = np.diff(np.sort(values))
    if degeneracy > 1:
        spacings = np.repeat(spacings[1::degeneracy], degeneracy)

    return spacings


def scale_support(support: Support, scale: float) -> Support:
    return scale * support[0], scale * support[1]


def truncated_polynomial_degrees(max_degree: int) -> range:
    return range(POLYNOMIAL_DEGREE_MIN, max_degree + 1, POLYNOMIAL_DEGREE_STEP)
