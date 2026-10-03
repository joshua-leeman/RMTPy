from collections.abc import Iterable, Iterator
from numbers import Integral
from pathlib import Path
from typing import cast, override

import attrs
import numpy as np

from ...compounds import CompoundEnsemble
from ...ensembles import RandomMatrixEnsemble
from ..base_data import Data
from ..base_plot import Plot
from ..base_simulation import DEFAULT_OUTPUT_ROOT, ExecutionState, Simulation
from ..statistics import REALIZATIONS_METADATA
from .transmission_coefficients import (
    TransmissionCoefficientsData,
    TransmissionCoefficientsPlot,
)
from .weisskopf_estimate import WeisskopfEstimateData, WeisskopfEstimatePlot

NUM_ENERGY_POINTS: int = 500


def load_transmission_coefficients_simulation(
    *,
    directory: str | Path,
) -> TransmissionCoefficientsSimulation:
    simulation = TransmissionCoefficientsSimulation.load(directory)
    if not isinstance(simulation, TransmissionCoefficientsSimulation):
        raise TypeError("Saved simulation is not a TransmissionCoefficientsSimulation.")

    return simulation


def plot_transmission_coefficients_simulation(*, directory: str | Path) -> None:
    simulation = load_transmission_coefficients_simulation(directory=directory)
    simulation.plot(directory)


def run_transmission_coefficients_simulation(
    *,
    compound: CompoundEnsemble,
    channel_indices: int | Iterable[int],
    realizs: int,
    directory: str | Path = DEFAULT_OUTPUT_ROOT,
) -> TransmissionCoefficientsSimulation:
    simulation = TransmissionCoefficientsSimulation(
        compound=compound,
        channel_indices=channel_indices,
        realizs=realizs,
    )
    simulation.execute()
    destination_directory = simulation.save(directory)
    plot_transmission_coefficients_simulation(directory=destination_directory)
    return simulation


def _normalize_channel_indices(channel_indices: object) -> tuple[int, ...]:
    if isinstance(channel_indices, np.ndarray) and channel_indices.ndim == 0:
        index_values = (channel_indices.item(),)
    elif np.isscalar(channel_indices):
        index_values = (channel_indices,)
    else:
        try:
            index_values = tuple(cast(Iterable[int], channel_indices))
        except TypeError as exc:
            raise TypeError(
                "`channel_indices` must be an integer or one-dimensional iterable "
                + "of integers."
            ) from exc

    normalized_indices: list[int] = []
    for channel_index in index_values:
        if isinstance(channel_index, (bool, np.bool_)) or not isinstance(
            channel_index, Integral
        ):
            raise TypeError("`channel_indices` must contain integer channel indices.")

        normalized_indices.append(int(channel_index))

    return tuple(normalized_indices)


def _validate_channel_indices(
    simulation: TransmissionCoefficientsSimulation,
    _: object,
    channel_indices: tuple[int, ...],
) -> None:
    if not channel_indices:
        raise ValueError("`channel_indices` must contain at least one channel index.")
    if len(set(channel_indices)) != len(channel_indices):
        raise ValueError("`channel_indices` must contain unique channel indices.")

    for channel_index in channel_indices:
        if not 0 <= channel_index < simulation.compound.num_channels:
            raise ValueError(
                "Channel index must be in "
                + f"[0, {simulation.compound.num_channels}), got {channel_index}."
            )


def _create_energy_grid(
    simulation: TransmissionCoefficientsSimulation,
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    spectral_energy_range = simulation.compound.ensemble.spectral_density.plot_range
    expanded_energy_range = tuple(1.5 * endpoint for endpoint in spectral_energy_range)
    energies = np.linspace(
        *cast(tuple[float, float], expanded_energy_range),
        NUM_ENERGY_POINTS,
        dtype=np.float64,
    )
    energies.flags.writeable = False
    return energies


def _create_transmission_coefficient_buffers(
    simulation: TransmissionCoefficientsSimulation,
) -> Iterable[TransmissionCoefficientsData]:
    transmission_coefficient_list: list[TransmissionCoefficientsData] = []
    for channel_index in simulation.channel_indices:
        transmission_coefficients = TransmissionCoefficientsData.create(
            energies=simulation.energies,
            channel_index=channel_index,
        )
        transmission_coefficient_list.append(transmission_coefficients)

    return tuple(transmission_coefficient_list)


def _compute_mean_level_spacings(
    simulation: TransmissionCoefficientsSimulation,
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    spectral_weight_density = np.asarray(
        simulation.compound.ensemble.spectral_density.weight_pdf(simulation.energies),
        dtype=np.float64,
    )
    if spectral_weight_density.shape != simulation.energies.shape:
        raise ValueError(
            "Spectral weight density must have shape "
            + f"{simulation.energies.shape}, got {spectral_weight_density.shape}."
        )

    mean_level_spacings = np.full(simulation.energies.shape, np.nan, dtype=np.float64)
    valid_density = np.isfinite(spectral_weight_density) & (spectral_weight_density > 0.0)
    mean_level_spacings[valid_density] = np.reciprocal(
        simulation.compound.ensemble.dimension * spectral_weight_density[valid_density]
    )
    return mean_level_spacings


def _create_weisskopf_estimate_buffer(
    simulation: TransmissionCoefficientsSimulation,
) -> WeisskopfEstimateData:
    return WeisskopfEstimateData.create(
        energies=simulation.energies,
        mean_level_spacings=_compute_mean_level_spacings(simulation),
        num_channels=simulation.compound.num_channels,
    )


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class TransmissionCoefficientsSimulation(Simulation):
    compound: CompoundEnsemble = attrs.field(
        converter=CompoundEnsemble.create,
    )
    channel_indices: tuple[int, ...] = attrs.field(
        converter=_normalize_channel_indices,
        validator=_validate_channel_indices,
    )
    realizs: int = attrs.field(
        converter=int,
        validator=attrs.validators.gt(0),
        metadata=REALIZATIONS_METADATA,
    )

    energies: np.ndarray[tuple[int], np.dtype[np.floating]] = attrs.field(
        default=attrs.Factory(_create_energy_grid, takes_self=True),
        init=False,
        repr=False,
    )
    transmission_coefficient_buffers: Iterable[TransmissionCoefficientsData] = (
        attrs.field(
            default=attrs.Factory(
                _create_transmission_coefficient_buffers,
                takes_self=True,
            ),
            repr=False,
        )
    )
    weisskopf_estimate_buffer: WeisskopfEstimateData = attrs.field(
        default=attrs.Factory(_create_weisskopf_estimate_buffer, takes_self=True),
        repr=False,
    )

    @override
    def __iter__(self) -> Iterator[Data]:
        yield from self.transmission_coefficient_buffers
        yield self.weisskopf_estimate_buffer

    @override
    def plot(self, directory: str | Path, /) -> None:
        if self.execution_state is not ExecutionState.COMPLETE:
            raise RuntimeError("A simulation may be plotted only after execution.")

        directory = Path(directory)
        for data in self:
            if not (directory / data.to_path).is_file():
                raise ValueError(f"Saved data `{data._file_name}` is missing.")

        for data in self:
            plot_cls: type[Plot]
            if isinstance(data, TransmissionCoefficientsData):
                plot_cls = TransmissionCoefficientsPlot
            elif isinstance(data, WeisskopfEstimateData):
                plot_cls = WeisskopfEstimatePlot
            else:
                raise TypeError(f"Data `{type(data).__name__}` has no plot class.")

            plot_cls(data=data, context=self.manifest).plot(
                directory / data.to_path.parent
            )

    @property
    @override
    def _rmg(self) -> RandomMatrixEnsemble:
        return self.compound.ensemble

    @property
    @override
    def _root_for_outputs(self) -> Path:
        return super()._root_for_outputs / self.compound.to_path

    def _realize_transmission_coefficients(self) -> None:
        expected_shape = (
            len(self.energies),
            self.compound.num_channels,
            self.compound.num_channels,
        )
        for scattering_matrices, _ in self.compound.scattering_matrix_stream(
            energies=self.energies,
            realizs=self.realizs,
        ):
            scattering_matrices = np.asarray(scattering_matrices)
            if scattering_matrices.shape != expected_shape:
                raise ValueError(
                    "Energy-resolved scattering matrices must have shape "
                    + f"{expected_shape}, got {scattering_matrices.shape}."
                )

            scattering_diagonal = np.diagonal(
                scattering_matrices,
                axis1=1,
                axis2=2,
            )
            for data, channel_index in zip(
                self.transmission_coefficient_buffers,
                self.channel_indices,
                strict=True,
            ):
                data.add_scattering_diagonal(scattering_diagonal[:, channel_index])

            self.weisskopf_estimate_buffer.add_scattering_diagonal(scattering_diagonal)

    @override
    def _execute(self) -> None:
        self._realize_transmission_coefficients()
        self._finalize()
