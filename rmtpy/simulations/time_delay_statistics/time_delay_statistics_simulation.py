from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import override

import attrs
import numpy as np

from ...compounds import CompoundEnsemble
from ...conversion import unwrap_json_value
from ...ensembles import RandomMatrixEnsemble
from ..base_data import Data
from ..base_simulation import DEFAULT_OUTPUT_ROOT, ExecutionState, Simulation
from ..statistics import REALIZATIONS_METADATA, truncated_polynomial_degree_range
from ..unfolding import CDF, TruncatedPolynomialCDFFactory, unfold_widths
from .time_delay_histogram import (
    TimeDelayHistogram,
    TimeDelayHistogramPlot,
    UnfoldedTimeDelayHistogramPlot,
)
from .time_delay_histogram.time_delay_histogram_plot import SpectralFormFactorsOverlay

ENERGIES_METADATA: dict[str, str] = {
    "latex_name": "E",
}


def load_time_delay_statistics_simulation(
    *,
    directory: str | Path,
) -> TimeDelayStatisticsSimulation:
    simulation = TimeDelayStatisticsSimulation.load(directory)

    return simulation


def plot_time_delay_statistics_simulation(
    *,
    directory: str | Path,
    spectral_statistics_directory: str | Path | None = None,
) -> None:
    simulation = load_time_delay_statistics_simulation(directory=directory)
    simulation.plot(
        directory,
        spectral_statistics_directory=spectral_statistics_directory,
    )


def run_time_delay_statistics_simulation(
    *,
    compound: CompoundEnsemble,
    energies: np.ndarray[tuple[int], np.dtype[np.floating]],
    realizs: int,
    directory: str | Path = DEFAULT_OUTPUT_ROOT,
    spectral_statistics_directory: str | Path | None = None,
) -> TimeDelayStatisticsSimulation:
    simulation = TimeDelayStatisticsSimulation(
        compound=compound,
        energies=energies,
        realizs=realizs,
    )
    simulation.execute()
    destination_directory = simulation.save(directory)
    plot_time_delay_statistics_simulation(
        directory=destination_directory,
        spectral_statistics_directory=spectral_statistics_directory,
    )
    return simulation


def _normalize_energies(
    energies: np.ndarray[tuple[int], np.dtype[np.floating]],
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    energies_array = np.array(energies, dtype=np.float64, copy=True, order="C")
    if energies_array.ndim == 0:
        energies_array = energies_array.reshape(1)
    if energies_array.ndim != 1:
        raise ValueError("`energies` must be a one-dimensional array-like.")
    if energies_array.size < 1:
        raise ValueError("`energies` must contain at least one value.")
    if not np.all(np.isfinite(energies_array)):
        raise ValueError("`energies` must contain finite values.")

    energies_array[energies_array == 0.0] = 0.0
    if np.unique(energies_array).size != energies_array.size:
        raise ValueError("`energies` must contain unique values.")

    energies_array.flags.writeable = False
    return energies_array


def unfold_time_delays(
    time_delays: np.ndarray[tuple[int], np.dtype[np.floating]],
    *,
    energy: float,
    cdf: CDF,
    dimension: int,
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    valid_time_delays = time_delays[np.isfinite(time_delays) & (time_delays > 0.0)]
    if valid_time_delays.size == 0:
        return valid_time_delays

    reciprocal_widths = np.reciprocal(valid_time_delays)
    unfolded_widths = unfold_widths(
        reciprocal_widths,
        centers=np.full_like(reciprocal_widths, energy),
        cdf=cdf,
        dimension=dimension,
    )
    valid_unfolded_widths = unfolded_widths[
        np.isfinite(unfolded_widths) & (unfolded_widths > 0.0)
    ]
    return np.reciprocal(valid_unfolded_widths)


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class TimeDelayStatisticsBuffers:
    polynomial_degree: int | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(
            (attrs.validators.instance_of(int), attrs.validators.gt(0)),
        ),
    )

    time_delays: tuple[TimeDelayHistogram, ...] = attrs.field(
        validator=attrs.validators.deep_iterable(
            member_validator=attrs.validators.instance_of(TimeDelayHistogram),
            iterable_validator=attrs.validators.instance_of(tuple),
        ),
        repr=False,
    )

    def __iter__(self) -> Iterator[Data]:
        yield from self.time_delays

    @classmethod
    def create_raw(
        cls,
        *,
        simulation: TimeDelayStatisticsSimulation,
    ) -> TimeDelayStatisticsBuffers:
        return TimeDelayStatisticsBuffers(
            time_delays=tuple(
                TimeDelayHistogram.create_raw(
                    ensemble=simulation.compound.ensemble,
                    energy_index=energy_index,
                    energy=float(energy),
                )
                for energy_index, energy in enumerate(simulation.energies)
            ),
        )

    @classmethod
    def create_unfolded(
        cls,
        *,
        simulation: TimeDelayStatisticsSimulation,
        unfolding: str,
        polynomial_degree: int | None = None,
    ) -> TimeDelayStatisticsBuffers:
        return TimeDelayStatisticsBuffers(
            polynomial_degree=polynomial_degree,
            time_delays=tuple(
                TimeDelayHistogram.create_unfolded(
                    ensemble=simulation.compound.ensemble,
                    energy_index=energy_index,
                    energy=float(energy),
                    unfolding=unfolding,
                    polynomial_degree=polynomial_degree,
                )
                for energy_index, energy in enumerate(simulation.energies)
            ),
        )

    def accumulate_raw_time_delays(
        self,
        time_delays: np.ndarray[tuple[int, int], np.dtype[np.floating]],
    ) -> None:
        for histogram, delay_values in zip(
            self.time_delays,
            time_delays,
            strict=True,
        ):
            histogram.add_histogram_contribution(delay_values)

    def accumulate_unfolded_time_delays(
        self,
        time_delays: np.ndarray[tuple[int, int], np.dtype[np.floating]],
        *,
        energies: np.ndarray[tuple[int], np.dtype[np.floating]],
        cdf: CDF,
        dimension: int,
    ) -> None:
        for energy, histogram, delay_values in zip(
            energies,
            self.time_delays,
            time_delays,
            strict=True,
        ):
            histogram.add_histogram_contribution(
                unfold_time_delays(
                    delay_values,
                    energy=float(energy),
                    cdf=cdf,
                    dimension=dimension,
                )
            )


def _create_raw_buffers(
    simulation: TimeDelayStatisticsSimulation,
) -> TimeDelayStatisticsBuffers:
    return TimeDelayStatisticsBuffers.create_raw(simulation=simulation)


def _create_weight_unfolded_buffers(
    simulation: TimeDelayStatisticsSimulation,
) -> TimeDelayStatisticsBuffers:
    return TimeDelayStatisticsBuffers.create_unfolded(
        simulation=simulation,
        unfolding="weight",
    )


def _create_averaged_unfolded_buffers(
    simulation: TimeDelayStatisticsSimulation,
) -> tuple[TimeDelayStatisticsBuffers, ...]:
    polynomial_degrees = truncated_polynomial_degree_range(
        max_degree=simulation.compound.ensemble.max_spectral_polynomial_degree
    )

    list_of_buffers: list[TimeDelayStatisticsBuffers] = []
    for polynomial_degree in polynomial_degrees:
        list_of_buffers.append(
            TimeDelayStatisticsBuffers.create_unfolded(
                simulation=simulation,
                unfolding="average",
                polynomial_degree=polynomial_degree,
            )
        )

    return tuple(list_of_buffers)


def _create_variate_unfolded_buffers(
    simulation: TimeDelayStatisticsSimulation,
) -> tuple[TimeDelayStatisticsBuffers, ...]:
    polynomial_degrees = truncated_polynomial_degree_range(
        max_degree=simulation.compound.ensemble.max_spectral_polynomial_degree
    )

    list_of_buffers: list[TimeDelayStatisticsBuffers] = []
    for polynomial_degree in polynomial_degrees:
        list_of_buffers.append(
            TimeDelayStatisticsBuffers.create_unfolded(
                simulation=simulation,
                unfolding="variate",
                polynomial_degree=polynomial_degree,
            )
        )

    return tuple(list_of_buffers)


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class TimeDelayStatisticsSimulation(Simulation):
    compound: CompoundEnsemble = attrs.field(
        converter=CompoundEnsemble.create,
    )
    energies: np.ndarray[tuple[int], np.dtype[np.floating]] = attrs.field(
        converter=_normalize_energies,
        metadata=ENERGIES_METADATA,
    )
    realizs: int = attrs.field(
        converter=int,
        validator=attrs.validators.gt(0),
        metadata=REALIZATIONS_METADATA,
    )

    raw_buffers: TimeDelayStatisticsBuffers = attrs.field(
        default=attrs.Factory(_create_raw_buffers, takes_self=True),
        repr=False,
    )
    wgt_unfolded_buffers: TimeDelayStatisticsBuffers = attrs.field(
        default=attrs.Factory(_create_weight_unfolded_buffers, takes_self=True),
        repr=False,
    )
    ave_unfolded_buffers: Iterable[TimeDelayStatisticsBuffers] = attrs.field(
        default=attrs.Factory(_create_averaged_unfolded_buffers, takes_self=True),
        repr=False,
    )
    var_unfolded_buffers: Iterable[TimeDelayStatisticsBuffers] = attrs.field(
        default=attrs.Factory(_create_variate_unfolded_buffers, takes_self=True),
        repr=False,
    )

    @override
    def __iter__(self) -> Iterator[Data]:
        yield from self.raw_buffers
        yield from self.wgt_unfolded_buffers

        for averaged_unfolded_buffers in self.ave_unfolded_buffers:
            yield from averaged_unfolded_buffers

        for variate_unfolded_buffers in self.var_unfolded_buffers:
            yield from variate_unfolded_buffers

    @override
    def plot(
        self,
        directory: str | Path,
        /,
        *,
        spectral_statistics_directory: str | Path | None = None,
    ) -> None:
        if self.execution_state is not ExecutionState.COMPLETE:
            raise RuntimeError("A simulation may be plotted only after execution.")

        directory = Path(directory)
        for data in self:
            if not (directory / data.to_path).is_file():
                raise ValueError(f"Saved data `{data._file_name}` is missing.")

        spectral_form_factors = None
        if spectral_statistics_directory is not None:
            spectral_form_factors = SpectralFormFactorsOverlay.from_directory(
                directory=spectral_statistics_directory,
                ensemble=self.compound.ensemble,
            )
            for data in self:
                if not isinstance(data, TimeDelayHistogram):
                    raise TypeError(f"Data `{type(data).__name__}` has no plot class.")
                _ = spectral_form_factors.validate(data)

        for data in self:
            unfolded = data.metadata.get("unfolding", "raw") != "raw"
            if isinstance(data, TimeDelayHistogram):
                plot = (
                    UnfoldedTimeDelayHistogramPlot(
                        data=data,
                        context=self.manifest,
                        spectral_form_factors=spectral_form_factors,
                    )
                    if unfolded
                    else TimeDelayHistogramPlot(
                        data=data,
                        context=self.manifest,
                        spectral_form_factors=spectral_form_factors,
                    )
                )
            else:
                raise TypeError(f"Data `{type(data).__name__}` has no plot class.")

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
            density=self.compound.ensemble.spectral_density,
            degrees=truncated_polynomial_degree_range(
                max_degree=self.compound.ensemble.max_spectral_polynomial_degree
            ),
            density_name="spectral",
        )

    def _realize_time_delay_statistics(self) -> None:
        ensemble = self.compound.ensemble
        spectral_density = ensemble.spectral_density

        cdf_factory = self._build_cdf_factory()

        average_cdfs = cdf_factory.average_interpolators()

        for time_delays, closed_eigenvalues in self.compound.time_delays_stream(
            energies=self.energies,
            realizs=self.realizs,
        ):
            time_delays = np.asarray(time_delays)
            expected_shape = (len(self.energies), self.compound.num_channels)
            if time_delays.shape != expected_shape:
                raise ValueError(
                    f"Time delays must have shape {expected_shape}, got "
                    + f"{time_delays.shape}."
                )

            self.raw_buffers.accumulate_raw_time_delays(time_delays)

            self.wgt_unfolded_buffers.accumulate_unfolded_time_delays(
                time_delays,
                energies=self.energies,
                cdf=spectral_density.weight_cdf,
                dimension=ensemble.dimension,
            )

            for cdf, buffers in zip(
                average_cdfs,
                self.ave_unfolded_buffers,
                strict=True,
            ):
                buffers.accumulate_unfolded_time_delays(
                    time_delays,
                    energies=self.energies,
                    cdf=cdf,
                    dimension=ensemble.dimension,
                )

            variate_coefficients = spectral_density.compute_variate_coeffs(
                closed_eigenvalues
            )
            variate_cdfs = cdf_factory.interpolators_from_coeffs(variate_coefficients)
            for cdf, buffers in zip(
                variate_cdfs,
                self.var_unfolded_buffers,
                strict=True,
            ):
                buffers.accumulate_unfolded_time_delays(
                    time_delays,
                    energies=self.energies,
                    cdf=cdf,
                    dimension=ensemble.dimension,
                )

    @override
    def _restore_execution(self) -> None:
        calibration = unwrap_json_value(self.manifest.execution.get("calibration"))
        if not isinstance(calibration, dict):
            raise ValueError("Saved spectral calibration is malformed.")
        if not calibration:
            return

        average_coefficients = np.asarray(
            calibration["average_coefficients"],
            dtype=self.compound.ensemble.real_dtype,
        )
        expected_shape = (self.compound.ensemble.max_spectral_polynomial_degree + 1,)
        if average_coefficients.shape != expected_shape:
            raise ValueError("Saved spectral calibration has an invalid shape.")

        object.__setattr__(
            self.compound.ensemble.spectral_density,
            "average_coeffs",
            average_coefficients,
        )

    @override
    def _execute(self) -> None:
        spectral_density = self.compound.ensemble.spectral_density
        has_average_coefficients = spectral_density.has_average_coeffs

        self._realize_time_delay_statistics()
        self._finalize()

        calibration: dict[str, object] = {}
        if spectral_density.has_average_coeffs:
            calibration = {
                "density": "spectral",
                "average_coefficients": spectral_density.average_coeffs,
                "timing": (
                    "previously_cached"
                    if has_average_coefficients
                    else "cached_during_execution"
                ),
            }

        self.manifest.execution["calibration"] = calibration
