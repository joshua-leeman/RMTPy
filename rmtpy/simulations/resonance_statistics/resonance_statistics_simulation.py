from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import override

import attrs
import numpy as np

from ...compounds import CompoundEnsemble
from ...conversion import unwrap_json_value
from ...ensembles import ManyBodyEnsemble, RandomMatrixEnsemble
from ..base_data import Data
from ..base_plot import Plot
from ..base_simulation import DEFAULT_OUTPUT_ROOT, ExecutionState, Simulation
from ..statistics import (
    REALIZATIONS_METADATA,
    nearest_neighbor_spacings,
    truncated_polynomial_degree_range,
)
from ..unfolding import CDF, TruncatedPolynomialCDFFactory, unfold_values, unfold_widths
from .complex_energy_histogram import (
    ComplexEnergyHistogram,
    ComplexEnergyHistogramPlot,
    UnfoldedComplexEnergyHistogramPlot,
)
from .resonance_coefficients_histogram import (
    ResonanceCoefficientsHistogram,
    ResonanceCoefficientsHistogramPlot,
)
from .resonance_form_factors import (
    FormFactorsData,
    ResonanceFormFactorsPlot,
    UnfoldedResonanceFormFactorsPlot,
)
from .resonance_histogram import (
    ResonanceHistogram,
    ResonanceHistogramPlot,
    UnfoldedResonanceHistogramPlot,
)
from .resonance_spacing_histogram import (
    ResonanceSpacingHistogram,
    ResonanceSpacingHistogramPlot,
    UnfoldedResonanceSpacingHistogramPlot,
)
from .width_histogram import (
    UnfoldedWidthHistogramPlot,
    WidthHistogram,
    WidthHistogramPlot,
)


def load_resonance_statistics_simulation(
    *,
    directory: str | Path,
) -> ResonanceStatisticsSimulation:
    simulation = ResonanceStatisticsSimulation.load(directory)
    if not isinstance(simulation, ResonanceStatisticsSimulation):
        raise TypeError("Saved simulation is not a ResonanceStatisticsSimulation.")

    return simulation


def plot_resonance_statistics_simulation(*, directory: str | Path) -> None:
    simulation = load_resonance_statistics_simulation(directory=directory)
    simulation.plot(directory)


def run_resonance_statistics_simulation(
    *,
    compound: CompoundEnsemble,
    realizs: int,
    directory: str | Path = DEFAULT_OUTPUT_ROOT,
) -> ResonanceStatisticsSimulation:
    simulation = ResonanceStatisticsSimulation(compound=compound, realizs=realizs)
    simulation.execute()
    destination_directory = simulation.save(directory)
    plot_resonance_statistics_simulation(directory=destination_directory)
    return simulation


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class ResonanceStatisticsBuffers:
    polynomial_degree: int | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(
            (attrs.validators.instance_of(int), attrs.validators.gt(0)),
        ),
    )

    resonance_centers: ResonanceHistogram = attrs.field(
        validator=attrs.validators.instance_of(ResonanceHistogram),
        repr=False,
    )
    resonance_widths: WidthHistogram = attrs.field(
        validator=attrs.validators.instance_of(WidthHistogram),
        repr=False,
    )
    nn_spacings: ResonanceSpacingHistogram = attrs.field(
        validator=attrs.validators.instance_of(ResonanceSpacingHistogram),
        repr=False,
    )
    complex_energies: ComplexEnergyHistogram = attrs.field(
        validator=attrs.validators.instance_of(ComplexEnergyHistogram),
        repr=False,
    )
    form_factors: FormFactorsData = attrs.field(
        validator=attrs.validators.instance_of(FormFactorsData),
        repr=False,
    )

    def __iter__(self) -> Iterator[Data]:
        yield self.resonance_centers
        yield self.resonance_widths
        yield self.nn_spacings
        yield self.complex_energies
        yield self.form_factors

    @classmethod
    def create_raw(
        cls,
        *,
        simulation: ResonanceStatisticsSimulation,
    ) -> ResonanceStatisticsBuffers:
        return ResonanceStatisticsBuffers(
            resonance_centers=ResonanceHistogram.create_raw(
                resonance_density=simulation.compound.resonance_density,
            ),
            resonance_widths=WidthHistogram.create_raw(),
            nn_spacings=ResonanceSpacingHistogram.create_raw(
                ensemble=simulation.compound.ensemble,
            ),
            complex_energies=ComplexEnergyHistogram.create_raw(),
            form_factors=FormFactorsData.create_raw(
                ensemble=simulation.compound.ensemble,
                file_name="resonance_form_factors",
            ),
        )

    @classmethod
    def create_unfolded(
        cls,
        *,
        simulation: ResonanceStatisticsSimulation,
        unfolding: str,
        polynomial_degree: int | None = None,
    ) -> ResonanceStatisticsBuffers:
        suffix = f"_degree_{polynomial_degree}" if polynomial_degree is not None else ""

        return ResonanceStatisticsBuffers(
            polynomial_degree=polynomial_degree,
            resonance_centers=ResonanceHistogram.create_unfolded(
                file_name=f"resonance_histogram_{unfolding}_unfolded{suffix}",
                dimension=simulation.compound.ensemble.dimension,
                unfolding=unfolding,
                polynomial_degree=polynomial_degree,
            ),
            resonance_widths=WidthHistogram.create_unfolded(
                file_name=f"width_histogram_{unfolding}_unfolded{suffix}",
                unfolding=unfolding,
                polynomial_degree=polynomial_degree,
            ),
            nn_spacings=ResonanceSpacingHistogram.create_unfolded(
                file_name=f"resonance_spacing_histogram_{unfolding}_unfolded{suffix}",
                unfolding=unfolding,
                polynomial_degree=polynomial_degree,
            ),
            complex_energies=ComplexEnergyHistogram.create_unfolded(
                file_name=f"complex_energy_histogram_{unfolding}_unfolded{suffix}",
                unfolding=unfolding,
                polynomial_degree=polynomial_degree,
            ),
            form_factors=FormFactorsData.create_unfolded(
                file_name=f"resonance_form_factors_{unfolding}_unfolded{suffix}",
                dimension=simulation.compound.ensemble.dimension,
                unfolding=unfolding,
                polynomial_degree=polynomial_degree,
            ),
        )

    def accumulate_raw_resonances(
        self,
        resonance_centers: np.ndarray[tuple[int], np.dtype[np.floating]],
        resonance_widths: np.ndarray[tuple[int], np.dtype[np.floating]],
        *,
        ensemble: ManyBodyEnsemble,
    ) -> None:
        self.resonance_centers.add_histogram_contribution(resonance_centers)
        self.resonance_widths.add_histogram_contribution(
            resonance_widths / ensemble.spectral_radius
        )

        nn_spacings = nearest_neighbor_spacings(
            resonance_centers,
            degeneracy=ensemble.eigval_degeneracy,
        )
        self.nn_spacings.add_histogram_contribution(nn_spacings)

        self.complex_energies.add_histogram_contribution(
            x_data=resonance_centers / ensemble.spectral_radius,
            y_data=resonance_widths / ensemble.spectral_radius,
        )

        self.form_factors.compute_moment_contributions(resonance_centers)

    def accumulate_unfolded_resonances(
        self,
        resonance_centers: np.ndarray[tuple[int], np.dtype[np.floating]],
        resonance_widths: np.ndarray[tuple[int], np.dtype[np.floating]],
        *,
        cdf: CDF,
        ensemble: ManyBodyEnsemble,
    ) -> None:
        unfolded_resonance_centers = unfold_values(
            resonance_centers,
            cdf=cdf,
            dimension=ensemble.dimension,
        )
        unfolded_resonance_widths = unfold_widths(
            resonance_widths,
            centers=resonance_centers,
            cdf=cdf,
            dimension=ensemble.dimension,
        )

        self.resonance_centers.add_histogram_contribution(unfolded_resonance_centers)
        self.resonance_widths.add_histogram_contribution(unfolded_resonance_widths)

        nn_spacings = nearest_neighbor_spacings(
            unfolded_resonance_centers,
            degeneracy=ensemble.eigval_degeneracy,
        )
        self.nn_spacings.add_histogram_contribution(nn_spacings)

        self.complex_energies.add_histogram_contribution(
            x_data=resonance_centers / ensemble.spectral_radius,
            y_data=unfolded_resonance_widths,
        )

        self.form_factors.compute_moment_contributions(unfolded_resonance_centers)


def _create_coefficient_buffers(
    simulation: ResonanceStatisticsSimulation,
) -> Iterable[ResonanceCoefficientsHistogram]:
    coefficient_histogram_list: list[ResonanceCoefficientsHistogram] = []
    for degree in range(
        1, simulation.compound.ensemble.max_spectral_polynomial_degree + 1
    ):
        coefficient_histogram = ResonanceCoefficientsHistogram.create(
            degree=degree,
            dimension=simulation.compound.ensemble.dimension,
        )
        coefficient_histogram_list.append(coefficient_histogram)

    return tuple(coefficient_histogram_list)


def _create_raw_buffers(
    simulation: ResonanceStatisticsSimulation,
) -> ResonanceStatisticsBuffers:
    return ResonanceStatisticsBuffers.create_raw(simulation=simulation)


def _create_weight_unfolded_buffers(
    simulation: ResonanceStatisticsSimulation,
) -> ResonanceStatisticsBuffers:
    return ResonanceStatisticsBuffers.create_unfolded(
        simulation=simulation,
        unfolding="weight",
    )


def _create_averaged_unfolded_buffers(
    simulation: ResonanceStatisticsSimulation,
) -> tuple[ResonanceStatisticsBuffers, ...]:
    polynomial_degrees = truncated_polynomial_degree_range(
        max_degree=simulation.compound.ensemble.max_spectral_polynomial_degree
    )

    list_of_buffers: list[ResonanceStatisticsBuffers] = []
    for polynomial_degree in polynomial_degrees:
        list_of_buffers.append(
            ResonanceStatisticsBuffers.create_unfolded(
                simulation=simulation,
                unfolding="average",
                polynomial_degree=polynomial_degree,
            )
        )

    return tuple(list_of_buffers)


def _create_variate_unfolded_buffers(
    simulation: ResonanceStatisticsSimulation,
) -> tuple[ResonanceStatisticsBuffers, ...]:
    polynomial_degrees = truncated_polynomial_degree_range(
        max_degree=simulation.compound.ensemble.max_spectral_polynomial_degree
    )

    list_of_buffers: list[ResonanceStatisticsBuffers] = []
    for polynomial_degree in polynomial_degrees:
        list_of_buffers.append(
            ResonanceStatisticsBuffers.create_unfolded(
                simulation=simulation,
                unfolding="variate",
                polynomial_degree=polynomial_degree,
            )
        )

    return tuple(list_of_buffers)


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class ResonanceStatisticsSimulation(Simulation):
    compound: CompoundEnsemble = attrs.field(
        converter=CompoundEnsemble.create,
    )
    realizs: int = attrs.field(
        converter=int,
        validator=attrs.validators.gt(0),
        metadata=REALIZATIONS_METADATA,
    )

    coefficient_buffers: Iterable[ResonanceCoefficientsHistogram] = attrs.field(
        default=attrs.Factory(_create_coefficient_buffers, takes_self=True),
        repr=False,
    )
    raw_buffers: ResonanceStatisticsBuffers = attrs.field(
        default=attrs.Factory(_create_raw_buffers, takes_self=True),
        repr=False,
    )
    wgt_unfolded_buffers: ResonanceStatisticsBuffers = attrs.field(
        default=attrs.Factory(_create_weight_unfolded_buffers, takes_self=True),
        repr=False,
    )
    ave_unfolded_buffers: Iterable[ResonanceStatisticsBuffers] = attrs.field(
        default=attrs.Factory(_create_averaged_unfolded_buffers, takes_self=True),
        repr=False,
    )
    var_unfolded_buffers: Iterable[ResonanceStatisticsBuffers] = attrs.field(
        default=attrs.Factory(_create_variate_unfolded_buffers, takes_self=True),
        repr=False,
    )

    @override
    def __iter__(self) -> Iterator[Data]:
        yield from self.coefficient_buffers
        yield from self.raw_buffers
        yield from self.wgt_unfolded_buffers

        for averaged_unfolded_buffers in self.ave_unfolded_buffers:
            yield from averaged_unfolded_buffers

        for variate_unfolded_buffers in self.var_unfolded_buffers:
            yield from variate_unfolded_buffers

    @override
    def plot(self, directory: str | Path, /) -> None:
        if self.execution_state is not ExecutionState.COMPLETE:
            raise RuntimeError("A simulation may be plotted only after execution.")

        directory = Path(directory)
        data_items = tuple(self)
        for data in data_items:
            if not (directory / data.to_path).is_file():
                raise ValueError(f"Saved data `{data._file_name}` is missing.")

        plots: list[Plot] = []
        for data in data_items:
            unfolded = data.metadata.get("unfolding", "raw") != "raw"
            plot_cls: type[Plot]
            if isinstance(data, ResonanceCoefficientsHistogram):
                plot_cls = ResonanceCoefficientsHistogramPlot
            elif isinstance(data, ResonanceHistogram):
                plot_cls = (
                    UnfoldedResonanceHistogramPlot if unfolded else ResonanceHistogramPlot
                )
            elif isinstance(data, WidthHistogram):
                plot_cls = UnfoldedWidthHistogramPlot if unfolded else WidthHistogramPlot
            elif isinstance(data, ResonanceSpacingHistogram):
                plot_cls = (
                    UnfoldedResonanceSpacingHistogramPlot
                    if unfolded
                    else ResonanceSpacingHistogramPlot
                )
            elif isinstance(data, ComplexEnergyHistogram):
                plot_cls = (
                    UnfoldedComplexEnergyHistogramPlot
                    if unfolded
                    else ComplexEnergyHistogramPlot
                )
            elif isinstance(data, FormFactorsData):
                plot_cls = (
                    UnfoldedResonanceFormFactorsPlot
                    if unfolded
                    else ResonanceFormFactorsPlot
                )
            else:
                raise TypeError(f"Data `{type(data).__name__}` has no plot class.")

            plots.append(plot_cls(data=data, context=self.manifest))

        coefficient_plots = tuple(
            plot for plot in plots if isinstance(plot, ResonanceCoefficientsHistogramPlot)
        )
        for plot in coefficient_plots:
            plot.set_derived_attributes()

        if coefficient_plots:
            widest_plot = max(
                coefficient_plots,
                key=lambda plot: plot.xlim[1] - plot.xlim[0],
            )

            for plot in coefficient_plots:
                plot.xlim = widest_plot.xlim
                plot.axes.xticks = widest_plot.axes.xticks
                plot.axes.xticks_minor = widest_plot.axes.xticks_minor
                plot.axes.xtick_labels = widest_plot.axes.xtick_labels

        for data, plot in zip(data_items, plots, strict=True):
            plot.plot(directory / data.to_path.parent)

    @property
    @override
    def _rmg(self) -> RandomMatrixEnsemble:
        return self.compound.ensemble

    @property
    @override
    def _root_for_outputs(self) -> Path:
        return super()._root_for_outputs / self.compound.to_path

    def _build_cdf_factory(self) -> TruncatedPolynomialCDFFactory:
        return TruncatedPolynomialCDFFactory(
            density=self.compound.resonance_density,
            degrees=truncated_polynomial_degree_range(
                max_degree=self.compound.ensemble.max_spectral_polynomial_degree
            ),
            density_name="resonance",
        )

    def _realize_resonance_statistics(self) -> None:
        ensemble = self.compound.ensemble
        resonance_density = self.compound.resonance_density
        coefficient_buffers = tuple(self.coefficient_buffers)
        cdf_factory = self._build_cdf_factory()

        average_cdfs = cdf_factory.average_interpolators()

        for complex_energies in self.compound.resonances_stream(realizs=self.realizs):
            resonance_centers = complex_energies.real
            resonance_widths = -2 * complex_energies.imag

            self.raw_buffers.accumulate_raw_resonances(
                resonance_centers,
                resonance_widths,
                ensemble=ensemble,
            )

            variate_coefficients = resonance_density.compute_variate_coeffs(
                resonance_centers
            )
            for histogram, coefficient in zip(
                coefficient_buffers,
                variate_coefficients[1 : len(coefficient_buffers) + 1],
                strict=True,
            ):
                histogram.add_histogram_contribution(
                    np.array([coefficient], dtype=np.float64)
                )

            self.wgt_unfolded_buffers.accumulate_unfolded_resonances(
                resonance_centers,
                resonance_widths,
                cdf=resonance_density.weight_cdf,
                ensemble=ensemble,
            )

            for cdf, buffers in zip(
                average_cdfs,
                self.ave_unfolded_buffers,
                strict=True,
            ):
                buffers.accumulate_unfolded_resonances(
                    resonance_centers,
                    resonance_widths,
                    cdf=cdf,
                    ensemble=ensemble,
                )

            variate_cdfs = cdf_factory.interpolators_from_coeffs(variate_coefficients)
            for cdf, buffers in zip(
                variate_cdfs,
                self.var_unfolded_buffers,
                strict=True,
            ):
                buffers.accumulate_unfolded_resonances(
                    resonance_centers,
                    resonance_widths,
                    cdf=cdf,
                    ensemble=ensemble,
                )

    @override
    def _restore_execution(self) -> None:
        calibration = unwrap_json_value(self.manifest.execution.get("calibration"))
        if not isinstance(calibration, dict):
            raise ValueError("Saved resonance calibration is malformed.")
        if not calibration:
            return

        average_coefficients = np.asarray(
            calibration["average_coefficients"],
            dtype=self.compound.ensemble.real_dtype,
        )
        expected_shape = (self.compound.ensemble.max_spectral_polynomial_degree + 1,)
        if average_coefficients.shape != expected_shape:
            raise ValueError("Saved resonance calibration has an invalid shape.")

        object.__setattr__(
            self.compound.resonance_density,
            "average_coeffs",
            average_coefficients,
        )

    @override
    def _execute(self) -> None:
        resonance_density = self.compound.resonance_density
        has_average_coefficients = resonance_density.has_average_coeffs

        self._realize_resonance_statistics()
        self._finalize()

        calibration: dict[str, object] = {}
        if resonance_density.has_average_coeffs:
            calibration = {
                "density": "resonance",
                "average_coefficients": resonance_density.average_coeffs,
                "timing": (
                    "previously_cached"
                    if has_average_coefficients
                    else "cached_during_execution"
                ),
            }

        self.manifest.execution["calibration"] = calibration
