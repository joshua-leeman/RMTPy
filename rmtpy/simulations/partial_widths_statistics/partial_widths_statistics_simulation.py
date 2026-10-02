from collections.abc import Iterable, Iterator
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
from .partial_width_histogram import PartialWidthHistogram, PartialWidthHistogramPlot
from .total_width_histogram import TotalWidthHistogram, TotalWidthHistogramPlot

type WidthHistogram = PartialWidthHistogram | TotalWidthHistogram


def load_partial_widths_statistics_simulation(
    *,
    directory: str | Path,
) -> PartialWidthsStatisticsSimulation:
    simulation = PartialWidthsStatisticsSimulation.load(directory)
    if not isinstance(simulation, PartialWidthsStatisticsSimulation):
        raise TypeError("Saved simulation is not a PartialWidthsStatisticsSimulation.")

    return simulation


def plot_partial_widths_statistics_simulation(*, directory: str | Path) -> None:
    simulation = load_partial_widths_statistics_simulation(directory=directory)
    simulation.plot(directory)


def run_partial_widths_statistics_simulation(
    *,
    compound: CompoundEnsemble,
    width_indices: Iterable[Iterable[int]],
    realizs: int,
    directory: str | Path = DEFAULT_OUTPUT_ROOT,
) -> PartialWidthsStatisticsSimulation:
    simulation = PartialWidthsStatisticsSimulation(
        compound=compound,
        width_indices=width_indices,
        realizs=realizs,
    )
    simulation.execute()
    destination_directory = simulation.save(directory)
    plot_partial_widths_statistics_simulation(directory=destination_directory)
    return simulation


def _normalize_width_indices(
    width_indices: Iterable[Iterable[int]],
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(index) for index in width_index) for width_index in width_indices
    )


def _validate_width_indices(
    simulation: PartialWidthsStatisticsSimulation,
    _: object,
    width_indices: tuple[tuple[int, ...], ...],
) -> None:
    if not width_indices:
        raise ValueError("`width_indices` must contain at least one selection.")

    if len(set(width_indices)) != len(width_indices):
        raise ValueError("`width_indices` must contain unique selections.")

    dimension = simulation.compound.ensemble.dimension
    num_channels = simulation.compound.num_channels

    for width_index in width_indices:
        if len(width_index) not in (1, 2):
            raise ValueError(
                "Each width index must be `(state,)` for a total width or "
                + "`(state, channel)` for a partial width."
            )

        state_index = width_index[0]
        if not 0 <= state_index < dimension:
            raise ValueError(
                f"Width state index must be in [0, {dimension}), got {state_index}."
            )

        if len(width_index) == 2:
            channel_index = width_index[1]
            if not 0 <= channel_index < num_channels:
                raise ValueError(
                    "Width channel index must be in "
                    + f"[0, {num_channels}), got {channel_index}."
                )


def _create_width_histogram_buffers(
    simulation: PartialWidthsStatisticsSimulation,
) -> Iterable[WidthHistogram]:
    width_histogram_list: list[WidthHistogram] = []
    for width_index in simulation.width_indices:
        if len(width_index) == 2:
            width_histogram = PartialWidthHistogram.create(
                state_index=width_index[0],
                channel_index=width_index[1],
            )
        elif len(width_index) == 1:
            width_histogram = TotalWidthHistogram.create(
                state_index=width_index[0],
            )
        else:
            raise ValueError("Invalid width index in `width_indices`.")

        width_histogram_list.append(width_histogram)

    return tuple(width_histogram_list)


def _compute_width(
    partial_widths: np.ndarray[tuple[int, int], np.dtype[np.floating]],
    *,
    width_index: tuple[int, ...],
) -> float:
    if len(width_index) == 2:
        return float(cast(np.floating, partial_widths[width_index[0], width_index[1]]))

    if len(width_index) == 1:
        row = cast(
            np.ndarray[tuple[int, ...], np.dtype[np.floating]],
            partial_widths[width_index[0]],
        )
        return float(np.sum(row))

    raise ValueError("Invalid width index in `width_indices`.")


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class PartialWidthsStatisticsSimulation(Simulation):
    compound: CompoundEnsemble = attrs.field(
        converter=CompoundEnsemble.create,
    )
    width_indices: tuple[tuple[int, ...], ...] = attrs.field(
        converter=_normalize_width_indices,
        validator=_validate_width_indices,
    )
    realizs: int = attrs.field(
        converter=int,
        validator=attrs.validators.gt(0),
        metadata=REALIZATIONS_METADATA,
    )

    width_histogram_buffers: Iterable[WidthHistogram] = attrs.field(
        default=attrs.Factory(_create_width_histogram_buffers, takes_self=True),
        repr=False,
    )

    @override
    def __iter__(self) -> Iterator[Data]:
        yield from self.width_histogram_buffers

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
            if isinstance(data, PartialWidthHistogram):
                plot_cls = PartialWidthHistogramPlot
            elif isinstance(data, TotalWidthHistogram):
                plot_cls = TotalWidthHistogramPlot
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

    def _realize_partial_widths_statistics(self) -> None:
        for partial_widths in self.compound.partial_widths_stream(realizs=self.realizs):
            for histogram in self.width_histogram_buffers:
                width_index = tuple(cast(Iterable[int], histogram.metadata["index"]))
                width = _compute_width(partial_widths, width_index=width_index)

                histogram.add_histogram_contribution(np.array([width], dtype=np.float64))
                width_sum = cast(float, histogram.metadata["average_width"])
                histogram.attach_metadata({"average_width": width_sum + width})

    def _finalize(self) -> None:
        average_widths = tuple(
            cast(float, histogram.metadata["average_width"]) / self.realizs
            for histogram in self.width_histogram_buffers
        )

        for histogram, average_width in zip(
            self.width_histogram_buffers,
            average_widths,
            strict=True,
        ):
            width_index = tuple(cast(Iterable[int], histogram.metadata["index"]))
            if not np.isfinite(average_width) or average_width <= 0.0:
                raise ValueError(
                    f"Average width for index {width_index} must be positive and finite."
                )

        for histogram, average_width in zip(
            self.width_histogram_buffers,
            average_widths,
            strict=True,
        ):
            histogram.attach_metadata({"average_width": average_width})
            histogram.bins[:] /= average_width
            histogram.compute_histogram()

    @override
    def _execute(self) -> None:
        self._realize_partial_widths_statistics()
        self._finalize()
