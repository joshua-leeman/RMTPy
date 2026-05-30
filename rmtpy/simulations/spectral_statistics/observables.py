from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import jn_zeros

from ..observable import Observable
from ..statistics import (
    create_coefficient_histograms,
    create_degree_observables,
    create_histogram_observable,
    create_observable,
    scale_support,
    truncated_polynomial_degrees,
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

if TYPE_CHECKING:
    from .spectral_statistics_simulation import SpectralStatisticsSimulation

SPECTRAL_COEFFICIENT_SUPPORT_DEFAULT: tuple[float, float] = (-0.2, 0.2)
UNFOLDED_LEVEL_SUPPORT_UNITS_DIMENSION_DEFAULT: tuple[float, float] = (-1.2, 1.2)
SPACING_SUPPORT_UNITS_MEAN_DEFAULT: tuple[float, float] = (0.0, 4.0)
SFF_LOG_D_TIME_SUPPORT_DEFAULT: tuple[float, float] = (-0.5, 1.5)
UNFOLDED_SFF_LOG_D_TIME_SUPPORT_DEFAULT: tuple[float, float] = (-1.5, 0.5)


def iterate_truncated_spectral_polynomial_degrees(
    simulation: SpectralStatisticsSimulation,
) -> range:
    return truncated_polynomial_degrees(
        simulation.ensemble.max_spectral_polynomial_degree
    )


def create_spectral_coeff_histograms(
    simulation: SpectralStatisticsSimulation,
    *,
    unfolding: str = "raw",
) -> list[Observable]:
    return create_coefficient_histograms(
        prefix="spectral",
        max_degree=simulation.ensemble.max_spectral_polynomial_degree,
        support=SPECTRAL_COEFFICIENT_SUPPORT_DEFAULT,
        plot_cls=SpectralCoefficientHistogramPlot,
        unfolding=unfolding,
    )


def create_spectral_histogram_observable(
    simulation: SpectralStatisticsSimulation,
    *,
    unfolding: str = "raw",
) -> Observable:
    return create_histogram_observable(
        file_name="spectral_histogram",
        support=simulation.ensemble.spectral_density.plot_range,
        plot_cls=SpectralHistogramPlot,
        metadata={"unfolding": unfolding},
    )


def create_unfolded_spectral_histogram_observable(
    simulation: SpectralStatisticsSimulation,
    file_name: str,
    *,
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
            simulation.ensemble.dimension,
        ),
        plot_cls=UnfoldedSpectralHistogramPlot,
        metadata=metadata,
    )


def create_weight_unfolded_spectral_histogram_observable(
    simulation: SpectralStatisticsSimulation,
) -> Observable:
    return create_unfolded_spectral_histogram_observable(
        simulation,
        file_name="spectral_histogram_weight_unfolded",
        unfolding="weight",
    )


def create_spacings_histogram_observable(
    simulation: SpectralStatisticsSimulation,
    *,
    unfolding: str = "raw",
) -> Observable:
    global_mean_spacing: float = (
        2 * simulation.ensemble.spectral_radius / simulation.ensemble.dimension
    )
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


def create_unfolded_spacings_histogram_observable(
    file_name: str,
    *,
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


def create_weight_unfolded_spacings_histogram_observable(
    _: SpectralStatisticsSimulation,
) -> Observable:
    return create_unfolded_spacings_histogram_observable(
        file_name="spacings_histogram_weight_unfolded",
        unfolding="weight",
    )


def create_spectral_form_factors_observable(
    simulation: SpectralStatisticsSimulation,
    *,
    unfolding: str = "raw",
) -> Observable:
    j_1_1: float = float(jn_zeros(1, 1)[0])
    return create_observable(
        data=FormFactorsData(
            file_name="spectral_form_factors",
            dimension=simulation.ensemble.dimension,
            logD_time_support=SFF_LOG_D_TIME_SUPPORT_DEFAULT,
            scale=j_1_1 / simulation.ensemble.spectral_radius,
        ),
        plot_cls=FormFactorsPlot,
        metadata={"unfolding": unfolding},
        finalize=finalize_form_factors,
    )


def create_unfolded_spectral_form_factors_observable(
    simulation: SpectralStatisticsSimulation,
    file_name: str,
    *,
    unfolding: str,
    degree: int | None = None,
) -> Observable:
    metadata: dict[str, int | str] = {"unfolding": unfolding}
    if degree is not None:
        metadata["degree"] = degree

    return create_observable(
        data=FormFactorsData(
            file_name=file_name,
            dimension=simulation.ensemble.dimension,
            logD_time_support=UNFOLDED_SFF_LOG_D_TIME_SUPPORT_DEFAULT,
            scale=2 * np.pi,
        ),
        plot_cls=UnfoldedFormFactorsPlot,
        metadata=metadata,
        finalize=finalize_form_factors,
    )


def create_weight_unfolded_spectral_form_factors_observable(
    simulation: SpectralStatisticsSimulation,
) -> Observable:
    return create_unfolded_spectral_form_factors_observable(
        simulation,
        file_name="spectral_form_factors_weight_unfolded",
        unfolding="weight",
    )


def create_avg_unfolded_spectral_histograms(
    simulation: SpectralStatisticsSimulation,
) -> list[Observable]:
    return create_degree_observables(
        degrees=iterate_truncated_spectral_polynomial_degrees(simulation),
        file_name_template="spectral_histogram_avg_unfolded_degree_{degree}",
        factory=lambda file_name, degree: create_unfolded_spectral_histogram_observable(
            simulation,
            file_name=file_name,
            unfolding="avg",
            degree=degree,
        ),
    )


def create_var_unfolded_spectral_histograms(
    simulation: SpectralStatisticsSimulation,
) -> list[Observable]:
    return create_degree_observables(
        degrees=iterate_truncated_spectral_polynomial_degrees(simulation),
        file_name_template="spectral_histogram_var_unfolded_degree_{degree}",
        factory=lambda file_name, degree: create_unfolded_spectral_histogram_observable(
            simulation,
            file_name=file_name,
            unfolding="var",
            degree=degree,
        ),
    )


def create_avg_unfolded_spacings_histograms(
    simulation: SpectralStatisticsSimulation,
) -> list[Observable]:
    return create_degree_observables(
        degrees=iterate_truncated_spectral_polynomial_degrees(simulation),
        file_name_template="spacings_histogram_avg_unfolded_degree_{degree}",
        factory=lambda file_name, degree: create_unfolded_spacings_histogram_observable(
            file_name=file_name,
            unfolding="avg",
            degree=degree,
        ),
    )


def create_var_unfolded_spacings_histograms(
    simulation: SpectralStatisticsSimulation,
) -> list[Observable]:
    return create_degree_observables(
        degrees=iterate_truncated_spectral_polynomial_degrees(simulation),
        file_name_template="spacings_histogram_var_unfolded_degree_{degree}",
        factory=lambda file_name, degree: create_unfolded_spacings_histogram_observable(
            file_name=file_name,
            unfolding="var",
            degree=degree,
        ),
    )


def create_avg_unfolded_spectral_form_factors(
    simulation: SpectralStatisticsSimulation,
) -> list[Observable]:
    return create_degree_observables(
        degrees=iterate_truncated_spectral_polynomial_degrees(simulation),
        file_name_template="spectral_form_factors_avg_unfolded_degree_{degree}",
        factory=lambda file_name, degree: (
            create_unfolded_spectral_form_factors_observable(
                simulation,
                file_name=file_name,
                unfolding="avg",
                degree=degree,
            )
        ),
    )


def create_var_unfolded_spectral_form_factors(
    simulation: SpectralStatisticsSimulation,
) -> list[Observable]:
    return create_degree_observables(
        degrees=iterate_truncated_spectral_polynomial_degrees(simulation),
        file_name_template="spectral_form_factors_var_unfolded_degree_{degree}",
        factory=lambda file_name, degree: (
            create_unfolded_spectral_form_factors_observable(
                simulation,
                file_name=file_name,
                unfolding="var",
                degree=degree,
            )
        ),
    )
