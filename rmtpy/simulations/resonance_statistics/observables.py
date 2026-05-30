from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import jn_zeros

from ..observable import Observable
from ..spectral_statistics.spectral_form_factors import finalize_form_factors
from ..statistics import (
    create_coefficient_histograms,
    create_degree_observables,
    create_histogram2d_observable,
    create_histogram_observable,
    create_observable,
    scale_support,
    truncated_polynomial_degrees,
)
from .complex_energy_histogram import (
    ComplexEnergyHistogramPlot,
    UnfoldedComplexEnergyHistogramPlot,
)
from .resonance_coefficients import ResonanceCoefficientHistogramPlot
from .resonance_form_factors import (
    FormFactorsData,
    ResonanceFormFactorsPlot,
    UnfoldedResonanceFormFactorsPlot,
)
from .resonance_histogram import (
    ResonanceHistogramPlot,
    UnfoldedResonanceHistogramPlot,
)
from .resonance_spacing_histogram import (
    ResonanceSpacingHistogramPlot,
    UnfoldedResonanceSpacingHistogramPlot,
)
from .width_histogram import UnfoldedWidthHistogramPlot, WidthHistogramPlot

if TYPE_CHECKING:
    from .resonance_statistics_simulation import ResonanceStatisticsSimulation

SUPPORT_SCALE_FACTOR_DEFAULT: float = 1.2
RESONANCE_COEFFICIENT_SUPPORT_DEFAULT: tuple[float, float] = (-0.2, 0.2)
RESONANCE_SUPPORT_UNITS_SPECTRAL_RADIUS_DEFAULT: tuple[float, float] = (
    -SUPPORT_SCALE_FACTOR_DEFAULT,
    SUPPORT_SCALE_FACTOR_DEFAULT,
)
UNFOLDED_RESONANCE_SUPPORT_UNITS_DIMENSION_DEFAULT: tuple[float, float] = (
    -SUPPORT_SCALE_FACTOR_DEFAULT,
    SUPPORT_SCALE_FACTOR_DEFAULT,
)
RESONANCE_WIDTH_LOG10_SUPPORT_DEFAULT: tuple[float, float] = (-4.0, 4.0)
UNFOLDED_RESONANCE_WIDTH_LOG10_SUPPORT_DEFAULT: tuple[float, float] = (-4.0, 4.0)
RESONANCE_SPACING_SUPPORT_UNITS_MEAN_SPACING_DEFAULT: tuple[float, float] = (0.0, 4.0)
UNFOLDED_RESONANCE_SPACING_SUPPORT_UNITS_MEAN_SPACING_DEFAULT: tuple[float, float] = (
    0.0,
    4.0,
)
COMPLEX_ENERGY_REAL_SUPPORT_UNITS_SPECTRAL_RADIUS_DEFAULT: tuple[float, float]
COMPLEX_ENERGY_REAL_SUPPORT_UNITS_SPECTRAL_RADIUS_DEFAULT = (
    RESONANCE_SUPPORT_UNITS_SPECTRAL_RADIUS_DEFAULT
)
COMPLEX_ENERGY_WIDTH_LOG10_SUPPORT_DEFAULT: tuple[float, float] = (-8.0, 8.0)
UNFOLDED_COMPLEX_ENERGY_REAL_SUPPORT_UNITS_SPECTRAL_RADIUS_DEFAULT: tuple[float, float]
UNFOLDED_COMPLEX_ENERGY_REAL_SUPPORT_UNITS_SPECTRAL_RADIUS_DEFAULT = (
    RESONANCE_SUPPORT_UNITS_SPECTRAL_RADIUS_DEFAULT
)
UNFOLDED_COMPLEX_ENERGY_WIDTH_LOG10_SUPPORT_DEFAULT: tuple[float, float] = (
    -8.0,
    8.0,
)
RESONANCE_WIDTH_LOG_BASE_DEFAULT: float = 10.0
COMPLEX_ENERGY_NUM_BINS_DEFAULT: int = 400
RESONANCE_FORM_FACTOR_LOG_D_TIME_SUPPORT_DEFAULT: tuple[float, float] = (-0.5, 1.5)
UNFOLDED_RESONANCE_FORM_FACTOR_LOG_D_TIME_SUPPORT_DEFAULT: tuple[float, float] = (
    -1.5,
    0.5,
)


def iterate_truncated_spectral_polynomial_degrees(
    simulation: ResonanceStatisticsSimulation,
) -> range:
    return truncated_polynomial_degrees(
        simulation.compound.ensemble.max_spectral_polynomial_degree
    )


def create_resonance_coeff_histograms(
    simulation: ResonanceStatisticsSimulation,
    *,
    unfolding: str = "raw",
) -> list[Observable]:
    return create_coefficient_histograms(
        prefix="resonance",
        max_degree=simulation.compound.ensemble.max_spectral_polynomial_degree,
        support=RESONANCE_COEFFICIENT_SUPPORT_DEFAULT,
        plot_cls=ResonanceCoefficientHistogramPlot,
        unfolding=unfolding,
    )


def create_resonance_histogram_observable(
    simulation: ResonanceStatisticsSimulation,
    *,
    unfolding: str = "raw",
) -> Observable:
    return create_histogram_observable(
        file_name="resonance_histogram",
        support=scale_support(
            RESONANCE_SUPPORT_UNITS_SPECTRAL_RADIUS_DEFAULT,
            simulation.compound.ensemble.spectral_radius,
        ),
        plot_cls=ResonanceHistogramPlot,
        metadata={"unfolding": unfolding},
    )


def create_unfolded_resonance_histogram_observable(
    simulation: ResonanceStatisticsSimulation,
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
            UNFOLDED_RESONANCE_SUPPORT_UNITS_DIMENSION_DEFAULT,
            simulation.compound.ensemble.dimension,
        ),
        plot_cls=UnfoldedResonanceHistogramPlot,
        metadata=metadata,
    )


def create_weight_unfolded_resonance_histogram_observable(
    simulation: ResonanceStatisticsSimulation,
) -> Observable:
    return create_unfolded_resonance_histogram_observable(
        simulation,
        file_name="resonance_histogram_weight_unfolded",
        unfolding="weight",
    )


def create_width_histogram_observable(
    _: ResonanceStatisticsSimulation,
    *,
    unfolding: str = "raw",
) -> Observable:
    return create_histogram_observable(
        file_name="width_histogram",
        log_base=RESONANCE_WIDTH_LOG_BASE_DEFAULT,
        support=RESONANCE_WIDTH_LOG10_SUPPORT_DEFAULT,
        plot_cls=WidthHistogramPlot,
        metadata={"unfolding": unfolding},
    )


def create_unfolded_width_histogram_observable(
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
        log_base=RESONANCE_WIDTH_LOG_BASE_DEFAULT,
        support=UNFOLDED_RESONANCE_WIDTH_LOG10_SUPPORT_DEFAULT,
        plot_cls=UnfoldedWidthHistogramPlot,
        metadata=metadata,
    )


def create_weight_unfolded_width_histogram_observable(
    _: ResonanceStatisticsSimulation,
) -> Observable:
    return create_unfolded_width_histogram_observable(
        file_name="width_histogram_weight_unfolded",
        unfolding="weight",
    )


def create_resonance_spacing_histogram_observable(
    simulation: ResonanceStatisticsSimulation,
    *,
    unfolding: str = "raw",
) -> Observable:
    global_mean_spacing: float = (
        2 * simulation.compound.ensemble.spectral_radius
    ) / simulation.compound.ensemble.dimension

    return create_histogram_observable(
        file_name="resonance_spacing_histogram",
        support=scale_support(
            RESONANCE_SPACING_SUPPORT_UNITS_MEAN_SPACING_DEFAULT,
            global_mean_spacing,
        ),
        plot_cls=ResonanceSpacingHistogramPlot,
        metadata={
            "global_mean_spacing": global_mean_spacing,
            "unfolding": unfolding,
        },
    )


def create_unfolded_resonance_spacing_histogram_observable(
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
        support=UNFOLDED_RESONANCE_SPACING_SUPPORT_UNITS_MEAN_SPACING_DEFAULT,
        plot_cls=UnfoldedResonanceSpacingHistogramPlot,
        metadata=metadata,
    )


def create_weight_unfolded_resonance_spacing_histogram_observable(
    _: ResonanceStatisticsSimulation,
) -> Observable:
    return create_unfolded_resonance_spacing_histogram_observable(
        file_name="resonance_spacing_histogram_wgt_unfolded",
        unfolding="weight",
    )


def create_complex_energy_histogram_observable(
    _: ResonanceStatisticsSimulation,
    *,
    unfolding: str = "raw",
) -> Observable:
    return create_histogram2d_observable(
        file_name="complex_energy_histogram",
        x_support=COMPLEX_ENERGY_REAL_SUPPORT_UNITS_SPECTRAL_RADIUS_DEFAULT,
        x_num_bins=COMPLEX_ENERGY_NUM_BINS_DEFAULT,
        y_log_base=RESONANCE_WIDTH_LOG_BASE_DEFAULT,
        y_support=COMPLEX_ENERGY_WIDTH_LOG10_SUPPORT_DEFAULT,
        y_num_bins=COMPLEX_ENERGY_NUM_BINS_DEFAULT,
        plot_cls=ComplexEnergyHistogramPlot,
        metadata={"unfolding": unfolding},
    )


def create_unfolded_complex_energy_histogram_observable(
    file_name: str,
    *,
    unfolding: str,
    degree: int | None = None,
) -> Observable:
    metadata: dict[str, int | str] = {"unfolding": unfolding}
    if degree is not None:
        metadata["degree"] = degree

    return create_histogram2d_observable(
        file_name=file_name,
        x_support=UNFOLDED_COMPLEX_ENERGY_REAL_SUPPORT_UNITS_SPECTRAL_RADIUS_DEFAULT,
        x_num_bins=COMPLEX_ENERGY_NUM_BINS_DEFAULT,
        y_log_base=RESONANCE_WIDTH_LOG_BASE_DEFAULT,
        y_support=UNFOLDED_COMPLEX_ENERGY_WIDTH_LOG10_SUPPORT_DEFAULT,
        y_num_bins=COMPLEX_ENERGY_NUM_BINS_DEFAULT,
        plot_cls=UnfoldedComplexEnergyHistogramPlot,
        metadata=metadata,
    )


def create_weight_unfolded_complex_energy_histogram_observable(
    _: ResonanceStatisticsSimulation,
) -> Observable:
    return create_unfolded_complex_energy_histogram_observable(
        file_name="complex_energy_histogram_weight_unfolded",
        unfolding="weight",
    )


def create_resonance_form_factors_observable(
    simulation: ResonanceStatisticsSimulation,
    *,
    unfolding: str = "raw",
) -> Observable:
    j_1_1: float = float(jn_zeros(1, 1)[0])
    return create_observable(
        data=FormFactorsData(
            file_name="resonance_form_factors",
            dimension=simulation.compound.ensemble.dimension,
            logD_time_support=RESONANCE_FORM_FACTOR_LOG_D_TIME_SUPPORT_DEFAULT,
            scale=j_1_1 / simulation.compound.ensemble.spectral_radius,
        ),
        plot_cls=ResonanceFormFactorsPlot,
        metadata={"unfolding": unfolding},
        finalize=finalize_form_factors,
    )


def create_unfolded_resonance_form_factors_observable(
    simulation: ResonanceStatisticsSimulation,
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
            dimension=simulation.compound.ensemble.dimension,
            logD_time_support=UNFOLDED_RESONANCE_FORM_FACTOR_LOG_D_TIME_SUPPORT_DEFAULT,
            scale=2 * np.pi,
        ),
        plot_cls=UnfoldedResonanceFormFactorsPlot,
        metadata=metadata,
        finalize=finalize_form_factors,
    )


def create_weight_unfolded_resonance_form_factors_observable(
    simulation: ResonanceStatisticsSimulation,
) -> Observable:
    return create_unfolded_resonance_form_factors_observable(
        simulation,
        file_name="resonance_form_factors_wgt_unfolded",
        unfolding="weight",
    )


def create_avg_unfolded_resonance_histograms(
    simulation: ResonanceStatisticsSimulation,
) -> list[Observable]:
    return create_degree_observables(
        degrees=iterate_truncated_spectral_polynomial_degrees(simulation),
        file_name_template="resonance_histogram_avg_unfolded_degree_{degree}",
        factory=lambda file_name, degree: (
            create_unfolded_resonance_histogram_observable(
                simulation,
                file_name=file_name,
                unfolding="avg",
                degree=degree,
            )
        ),
    )


def create_var_unfolded_resonance_histograms(
    simulation: ResonanceStatisticsSimulation,
) -> list[Observable]:
    return create_degree_observables(
        degrees=iterate_truncated_spectral_polynomial_degrees(simulation),
        file_name_template="resonance_histogram_var_unfolded_degree_{degree}",
        factory=lambda file_name, degree: (
            create_unfolded_resonance_histogram_observable(
                simulation,
                file_name=file_name,
                unfolding="var",
                degree=degree,
            )
        ),
    )


def create_avg_unfolded_width_histograms(
    simulation: ResonanceStatisticsSimulation,
) -> list[Observable]:
    return create_degree_observables(
        degrees=iterate_truncated_spectral_polynomial_degrees(simulation),
        file_name_template="width_histogram_avg_unfolded_degree_{degree}",
        factory=lambda file_name, degree: create_unfolded_width_histogram_observable(
            file_name=file_name,
            unfolding="avg",
            degree=degree,
        ),
    )


def create_var_unfolded_width_histograms(
    simulation: ResonanceStatisticsSimulation,
) -> list[Observable]:
    return create_degree_observables(
        degrees=iterate_truncated_spectral_polynomial_degrees(simulation),
        file_name_template="width_histogram_var_unfolded_degree_{degree}",
        factory=lambda file_name, degree: create_unfolded_width_histogram_observable(
            file_name=file_name,
            unfolding="var",
            degree=degree,
        ),
    )


def create_avg_unfolded_resonance_spacing_histograms(
    simulation: ResonanceStatisticsSimulation,
) -> list[Observable]:
    return create_degree_observables(
        degrees=iterate_truncated_spectral_polynomial_degrees(simulation),
        file_name_template="resonance_spacing_histogram_avg_unfolded_degree_{degree}",
        factory=lambda file_name, degree: (
            create_unfolded_resonance_spacing_histogram_observable(
                file_name=file_name,
                unfolding="avg",
                degree=degree,
            )
        ),
    )


def create_var_unfolded_resonance_spacing_histograms(
    simulation: ResonanceStatisticsSimulation,
) -> list[Observable]:
    return create_degree_observables(
        degrees=iterate_truncated_spectral_polynomial_degrees(simulation),
        file_name_template="resonance_spacing_histogram_var_unfolded_degree_{degree}",
        factory=lambda file_name, degree: (
            create_unfolded_resonance_spacing_histogram_observable(
                file_name=file_name,
                unfolding="var",
                degree=degree,
            )
        ),
    )


def create_avg_unfolded_complex_energy_histograms(
    simulation: ResonanceStatisticsSimulation,
) -> list[Observable]:
    return create_degree_observables(
        degrees=iterate_truncated_spectral_polynomial_degrees(simulation),
        file_name_template="complex_energy_histogram_avg_unfolded_degree_{degree}",
        factory=lambda file_name, degree: (
            create_unfolded_complex_energy_histogram_observable(
                file_name=file_name,
                unfolding="avg",
                degree=degree,
            )
        ),
    )


def create_var_unfolded_complex_energy_histograms(
    simulation: ResonanceStatisticsSimulation,
) -> list[Observable]:
    return create_degree_observables(
        degrees=iterate_truncated_spectral_polynomial_degrees(simulation),
        file_name_template="complex_energy_histogram_var_unfolded_degree_{degree}",
        factory=lambda file_name, degree: (
            create_unfolded_complex_energy_histogram_observable(
                file_name=file_name,
                unfolding="var",
                degree=degree,
            )
        ),
    )


def create_avg_unfolded_resonance_form_factors(
    simulation: ResonanceStatisticsSimulation,
) -> list[Observable]:
    return create_degree_observables(
        degrees=iterate_truncated_spectral_polynomial_degrees(simulation),
        file_name_template="resonance_form_factors_avg_unfolded_degree_{degree}",
        factory=lambda file_name, degree: (
            create_unfolded_resonance_form_factors_observable(
                simulation,
                file_name=file_name,
                unfolding="avg",
                degree=degree,
            )
        ),
    )


def create_var_unfolded_resonance_form_factors(
    simulation: ResonanceStatisticsSimulation,
) -> list[Observable]:
    return create_degree_observables(
        degrees=iterate_truncated_spectral_polynomial_degrees(simulation),
        file_name_template="resonance_form_factors_var_unfolded_degree_{degree}",
        factory=lambda file_name, degree: (
            create_unfolded_resonance_form_factors_observable(
                simulation,
                file_name=file_name,
                unfolding="var",
                degree=degree,
            )
        ),
    )
