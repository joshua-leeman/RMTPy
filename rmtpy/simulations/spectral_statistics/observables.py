from __future__ import annotations

import numpy as np
from scipy.special import jn_zeros

import rmtpy.density
from rmtpy.ensembles import ManyBodyEnsemble

from ..observable import Observable
from ..statistics import (
    create_coefficient_histograms,
    create_histogram_observable,
    create_observable,
    scale_support,
)
from .spacings_histogram import (
    SpacingsHistogramPlot,
    UnfoldedSpacingsHistogramPlot,
)
from .spectral_coefficients import SpectralCoefficientHistogramPlot
from .spectral_form_factors import (
    FormFactorsData,
    FormFactorsPlot,
    UnfoldedFormFactorsPlot,
    finalize_form_factors,
)
from .spectral_histogram import (
    SpectralHistogramPlot,
    UnfoldedSpectralHistogramPlot,
)

SFF_LOG_D_TIME_SUPPORT_DEFAULT: tuple[float, float] = (-0.5, 1.5)

SPACING_SUPPORT_UNITS_MEAN_DEFAULT: tuple[float, float] = (0.0, 4.0)

SPECTRAL_COEFFICIENT_SUPPORT_DEFAULT: tuple[float, float] = (-0.2, 0.2)

UNFOLDED_LEVEL_SUPPORT_UNITS_DIMENSION_DEFAULT: tuple[float, float] = (-1.2, 1.2)

UNFOLDED_SFF_LOG_D_TIME_SUPPORT_DEFAULT: tuple[float, float] = (-1.5, 0.5)


def create_raw_spacings_histogram_observable(
    *,
    ensemble: ManyBodyEnsemble,
    unfolding: str = "raw",
) -> Observable:
    global_mean_spacing = 2 * ensemble.spectral_radius / ensemble.dimension
    return create_histogram_observable(
        file_name="spacings_histogram",
        support=scale_support(
            SPACING_SUPPORT_UNITS_MEAN_DEFAULT,
            global_mean_spacing,
        ),
        plot_cls=SpacingsHistogramPlot,
        metadata={
            "global_mean_spacing": global_mean_spacing,
            "unfolding": unfolding,
        },
    )


def create_raw_spectral_form_factors_observable(
    *,
    ensemble: ManyBodyEnsemble,
    unfolding: str = "raw",
) -> Observable:
    j_1_1 = float(jn_zeros(1, 1)[0])
    return create_observable(
        data=FormFactorsData(
            file_name="spectral_form_factors",
            dimension=ensemble.dimension,
            logD_time_support=SFF_LOG_D_TIME_SUPPORT_DEFAULT,
            scale=j_1_1 / ensemble.spectral_radius,
        ),
        plot_cls=FormFactorsPlot,
        metadata={"unfolding": unfolding},
        finalize=finalize_form_factors,
    )


def create_raw_spectral_histogram_observable(
    *,
    spectral_density: rmtpy.density.DensityModel,
    unfolding: str = "raw",
) -> Observable:
    return create_histogram_observable(
        file_name="spectral_histogram",
        support=spectral_density.plot_range,
        plot_cls=SpectralHistogramPlot,
        metadata={"unfolding": unfolding},
    )


def create_spectral_coeff_histograms(
    *,
    max_degree: int,
    unfolding: str = "raw",
) -> tuple[Observable, ...]:
    return tuple(
        create_coefficient_histograms(
            prefix="spectral",
            max_degree=max_degree,
            support=SPECTRAL_COEFFICIENT_SUPPORT_DEFAULT,
            plot_cls=SpectralCoefficientHistogramPlot,
            unfolding=unfolding,
        )
    )


def create_unfolded_spacings_histogram_observable(
    *,
    file_name: str,
    unfolding: str,
    degree: int | None = None,
) -> Observable:
    metadata: dict[str, int | str] = {"unfolding": unfolding}
    if degree is not None:
        metadata["degree"] = degree

    return create_histogram_observable(
        file_name=file_name,
        support=SPACING_SUPPORT_UNITS_MEAN_DEFAULT,
        plot_cls=UnfoldedSpacingsHistogramPlot,
        metadata=metadata,
    )


def create_unfolded_spectral_form_factors_observable(
    *,
    file_name: str,
    dimension: int,
    unfolding: str,
    degree: int | None = None,
) -> Observable:
    metadata: dict[str, int | str] = {"unfolding": unfolding}
    if degree is not None:
        metadata["degree"] = degree

    return create_observable(
        data=FormFactorsData(
            file_name=file_name,
            dimension=dimension,
            logD_time_support=UNFOLDED_SFF_LOG_D_TIME_SUPPORT_DEFAULT,
            scale=2 * np.pi,
        ),
        plot_cls=UnfoldedFormFactorsPlot,
        metadata=metadata,
        finalize=finalize_form_factors,
    )


def create_unfolded_spectral_histogram_observable(
    *,
    file_name: str,
    dimension: int,
    unfolding: str,
    degree: int | None = None,
) -> Observable:
    metadata: dict[str, int | str] = {"unfolding": unfolding}
    if degree is not None:
        metadata["degree"] = degree

    return create_histogram_observable(
        file_name=file_name,
        support=scale_support(
            UNFOLDED_LEVEL_SUPPORT_UNITS_DIMENSION_DEFAULT,
            dimension,
        ),
        plot_cls=UnfoldedSpectralHistogramPlot,
        metadata=metadata,
    )
