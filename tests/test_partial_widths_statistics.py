import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
from matplotlib import pyplot as plt

from rmtpy.compounds import CompoundEnsemble
from rmtpy.ensembles import GOE
from rmtpy.simulations.base_simulation import ExecutionState
from rmtpy.simulations.partial_widths_statistics import (
    PartialWidthsStatisticsSimulation,
    load_partial_widths_statistics_simulation,
    plot_partial_widths_statistics_simulation,
    run_partial_widths_statistics_simulation,
)
from rmtpy.simulations.partial_widths_statistics.partial_width_histogram import (
    PartialWidthHistogram,
    PartialWidthHistogramPlot,
)
from rmtpy.simulations.partial_widths_statistics.total_width_histogram import (
    TotalWidthHistogram,
    TotalWidthHistogramPlot,
)


def build_compound(
    *,
    num_free_complex_fermions: int = 1,
    seed: int = 123,
) -> CompoundEnsemble:
    return CompoundEnsemble(
        ensemble=GOE(num_majoranas=4, seed=seed),
        num_free_complex_fermions=num_free_complex_fermions,
    )


type SelectedWidthHistogram = PartialWidthHistogram | TotalWidthHistogram


def selected_width_histograms(
    simulation: PartialWidthsStatisticsSimulation,
) -> tuple[SelectedWidthHistogram, ...]:
    return cast(tuple[SelectedWidthHistogram, ...], tuple(simulation))


def histogram_counts(
    samples: tuple[float, ...],
    *,
    bins: np.ndarray[tuple[int], np.dtype[np.floating]],
) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
    counts = np.zeros(len(bins) - 1, dtype=np.int64)
    indices = np.searchsorted(bins, samples, side="right") - 1
    valid = (indices >= 0) & (indices < len(counts))
    np.add.at(counts, indices[valid], 1)
    return counts


class PartialWidthsStatisticsTests(unittest.TestCase):
    def test_plot_titles_preserve_width_indices_as_ordered_metadata(self) -> None:
        simulation = PartialWidthsStatisticsSimulation(
            compound=build_compound(num_free_complex_fermions=2),
            width_indices=((1, 0), (1,)),
            realizs=1,
        )
        partial_width, total_width = selected_width_histograms(simulation)
        ensemble = simulation.compound.ensemble.to_latex

        partial_plot = PartialWidthHistogramPlot(
            data=partial_width,
            context=simulation.manifest,
        )
        total_plot = TotalWidthHistogramPlot(
            data=total_width,
            context=simulation.manifest,
        )
        partial_plot.set_derived_attributes()
        total_plot.set_derived_attributes()

        self.assertEqual(
            partial_plot.axes.title,
            "Partial Width PDF: "
            + ensemble
            + r", $N_\text{f} = {2}$, $\mu = {1}$, $a = {0}$",
        )
        self.assertEqual(
            total_plot.axes.title,
            "Total Width PDF: " + ensemble + r", $N_\text{f} = {2}$, $\mu = {1}$",
        )
        self.assertNotIn("\n", partial_plot.axes.title)
        self.assertNotIn("\n", total_plot.axes.title)

    def test_required_width_indices_are_normalized_validated_and_ordered(self) -> None:
        simulation = PartialWidthsStatisticsSimulation(
            compound=build_compound(),
            width_indices=([1], [0, 0], [1, 1]),
            realizs=1,
        )

        self.assertEqual(simulation.width_indices, ((1,), (0, 0), (1, 1)))
        self.assertEqual(
            tuple(type(histogram) for histogram in simulation),
            (TotalWidthHistogram, PartialWidthHistogram, PartialWidthHistogram),
        )
        self.assertEqual(
            tuple(histogram._file_name for histogram in simulation),
            (
                "total_width_state_1_histogram",
                "partial_width_state_0_channel_0_histogram",
                "partial_width_state_1_channel_1_histogram",
            ),
        )
        self.assertEqual(
            tuple(
                tuple(cast(list[int], histogram.metadata["index"]))
                for histogram in simulation
            ),
            simulation.width_indices,
        )
        self.assertTrue(
            all(histogram.metadata["unfolding"] == "raw" for histogram in simulation)
        )

        compound = build_compound(num_free_complex_fermions=0)
        invalid_cases = (
            (),
            ((0, 0), (0, 0)),
            ((0, 0, 0),),
            ((2,),),
            ((0, 1),),
        )
        for width_indices in invalid_cases:
            with self.subTest(width_indices=width_indices), self.assertRaises(ValueError):
                PartialWidthsStatisticsSimulation(
                    compound=compound,
                    width_indices=width_indices,
                    realizs=1,
                )

        with self.assertRaises(TypeError):
            PartialWidthsStatisticsSimulation(  # pyright: ignore[reportCallIssue]
                compound=build_compound(),
                realizs=1,
            )

    def test_partial_and_total_widths_are_accumulated_and_finalized(self) -> None:
        simulation = PartialWidthsStatisticsSimulation(
            compound=build_compound(),
            width_indices=((0, 0), (1, 0), (1, 1), (0,), (1,)),
            realizs=2,
        )
        initial_bins = tuple(
            histogram.bins.copy() for histogram in selected_width_histograms(simulation)
        )
        samples = (
            np.array([[2.0, 4.0], [6.0, 8.0]]),
            np.array([[4.0, 8.0], [12.0, 16.0]]),
        )

        with patch.object(
            CompoundEnsemble,
            "partial_widths_stream",
            return_value=iter(samples),
        ):
            returned = simulation.execute()

        self.assertIsNone(returned)
        self.assertEqual(simulation.execution_state, ExecutionState.COMPLETE)

        raw_values = (
            (2.0, 4.0),
            (6.0, 12.0),
            (8.0, 16.0),
            (6.0, 12.0),
            (14.0, 28.0),
        )
        expected_means = (3.0, 9.0, 12.0, 9.0, 21.0)
        for histogram, bins, values, expected_mean in zip(
            selected_width_histograms(simulation),
            initial_bins,
            raw_values,
            expected_means,
            strict=True,
        ):
            expected_counts = histogram_counts(values, bins=bins)
            np.testing.assert_array_equal(histogram.counts, expected_counts)
            np.testing.assert_allclose(histogram.bins, bins / expected_mean)
            np.testing.assert_allclose(
                histogram.histogram,
                expected_counts / (np.sum(expected_counts) * np.diff(histogram.bins)),
            )
            self.assertEqual(histogram.realizs, 2)
            self.assertEqual(histogram.metadata["average_width"], expected_mean)

    def test_invalid_average_width_is_terminal_and_does_not_rescale_bins(
        self,
    ) -> None:
        for invalid_width in (0.0, np.nan, np.inf):
            with self.subTest(invalid_width=invalid_width):
                simulation = PartialWidthsStatisticsSimulation(
                    compound=build_compound(),
                    width_indices=((0, 0), (0,)),
                    realizs=1,
                )
                initial_bins = tuple(
                    histogram.bins.copy()
                    for histogram in selected_width_histograms(simulation)
                )
                sample = np.full((2, 2), invalid_width)

                with (
                    patch.object(
                        CompoundEnsemble,
                        "partial_widths_stream",
                        return_value=iter((sample,)),
                    ),
                    self.assertRaisesRegex(ValueError, "positive and finite"),
                ):
                    simulation.execute()

                self.assertEqual(simulation.execution_state, ExecutionState.FAILED)
                for histogram, bins in zip(
                    selected_width_histograms(simulation),
                    initial_bins,
                    strict=True,
                ):
                    np.testing.assert_array_equal(histogram.bins, bins)

                with self.assertRaisesRegex(RuntimeError, "only once"):
                    simulation.execute()
                with self.assertRaisesRegex(RuntimeError, "only after execution"):
                    simulation.save()
                with self.assertRaisesRegex(RuntimeError, "only after execution"):
                    simulation.plot(Path("unused"))

    def test_seeded_execution_matches_direct_stream_and_captures_rng(self) -> None:
        simulation = PartialWidthsStatisticsSimulation(
            compound=build_compound(seed=314159),
            width_indices=((0, 0), (1,)),
            realizs=2,
        )
        control = build_compound(seed=314159)
        initial_rng_state = deepcopy(simulation.compound.rng_state)
        initial_bins = tuple(
            histogram.bins.copy() for histogram in selected_width_histograms(simulation)
        )
        samples = tuple(control.partial_widths_stream(realizs=2))

        simulation.execute()

        self.assertEqual(simulation.compound.rng_state, control.rng_state)
        self.assertEqual(simulation.manifest.rng["initial_state"], initial_rng_state)
        self.assertEqual(simulation.manifest.rng["final_state"], control.rng_state)

        expected_values = (
            tuple(float(sample[0, 0]) for sample in samples),
            tuple(float(np.sum(sample[1])) for sample in samples),
        )
        for histogram, bins, values in zip(
            selected_width_histograms(simulation),
            initial_bins,
            expected_values,
            strict=True,
        ):
            np.testing.assert_array_equal(
                histogram.counts,
                histogram_counts(values, bins=bins),
            )
            self.assertEqual(
                histogram.metadata["average_width"],
                sum(values) / simulation.realizs,
            )

    def test_save_load_plot_dispatch_and_missing_data_validation(self) -> None:
        simulation = PartialWidthsStatisticsSimulation(
            compound=build_compound(seed=77),
            width_indices=((1,), (0, 0)),
            realizs=2,
        )
        simulation.execute()
        completed_rng_state = deepcopy(simulation.compound.rng_state)

        with tempfile.TemporaryDirectory() as temporary_directory:
            destination_directory = simulation.save(temporary_directory)
            restored_simulation = load_partial_widths_statistics_simulation(
                directory=destination_directory
            )

            self.assertEqual(
                restored_simulation.execution_state,
                ExecutionState.COMPLETE,
            )
            self.assertEqual(
                restored_simulation.width_indices,
                simulation.width_indices,
            )
            self.assertEqual(
                tuple(type(histogram) for histogram in restored_simulation),
                (TotalWidthHistogram, PartialWidthHistogram),
            )
            for restored_histogram, original_histogram in zip(
                selected_width_histograms(restored_simulation),
                selected_width_histograms(simulation),
                strict=True,
            ):
                np.testing.assert_array_equal(
                    restored_histogram.counts,
                    original_histogram.counts,
                )
                np.testing.assert_allclose(
                    restored_histogram.histogram,
                    original_histogram.histogram,
                )
                self.assertEqual(
                    restored_histogram.metadata,
                    original_histogram.metadata,
                )

            with (
                patch.object(
                    PartialWidthHistogramPlot,
                    "plot",
                    autospec=True,
                ) as partial_plot,
                patch.object(
                    TotalWidthHistogramPlot,
                    "plot",
                    autospec=True,
                ) as total_plot,
            ):
                plot_partial_widths_statistics_simulation(directory=destination_directory)

            partial_plot.assert_called_once()
            total_plot.assert_called_once()
            self.assertIsInstance(
                partial_plot.call_args.args[0].data, PartialWidthHistogram
            )
            self.assertIsInstance(total_plot.call_args.args[0].data, TotalWidthHistogram)
            self.assertIsNotNone(partial_plot.call_args.args[0].context)

            unexpected = PartialWidthHistogram.create(
                state_index=1,
                channel_index=1,
            )
            unexpected.save(directory=destination_directory)
            with self.assertRaisesRegex(ValueError, "not part of the simulation"):
                load_partial_widths_statistics_simulation(directory=destination_directory)
            (destination_directory / unexpected.to_path).unlink()

            missing_data = destination_directory / next(iter(restored_simulation)).to_path
            missing_data.unlink()
            with self.assertRaisesRegex(ValueError, "Saved data .* is missing"):
                load_partial_widths_statistics_simulation(directory=destination_directory)
            with self.assertRaisesRegex(ValueError, "Saved data .* is missing"):
                restored_simulation.plot(destination_directory)

        self.assertEqual(simulation.compound.rng_state, completed_rng_state)

    def test_plot_configuration_uses_a_detached_compound(self) -> None:
        simulation = PartialWidthsStatisticsSimulation(
            compound=build_compound(seed=902),
            width_indices=((0, 0), (1,)),
            realizs=1,
        )
        simulation.execute()
        completed_rng_state = deepcopy(simulation.compound.rng_state)
        partial_histogram, total_histogram = tuple(simulation)

        partial_plot = PartialWidthHistogramPlot(
            data=partial_histogram,
            context=simulation.manifest,
        )
        total_plot = TotalWidthHistogramPlot(
            data=total_histogram,
            context=simulation.manifest,
        )
        try:
            with (
                patch.object(PartialWidthHistogramPlot, "finish_plot") as partial_finish,
                patch.object(TotalWidthHistogramPlot, "finish_plot") as total_finish,
            ):
                partial_plot.plot(Path("unused"))
                total_plot.plot(Path("unused"))

            partial_finish.assert_called_once_with(path=Path("unused"))
            total_finish.assert_called_once_with(path=Path("unused"))
        finally:
            plt.close("all")

        self.assertIsNot(partial_plot.compound, simulation.compound)
        self.assertIsNot(total_plot.compound, simulation.compound)
        self.assertEqual(simulation.compound.rng_state, completed_rng_state)

    def test_run_helper_executes_saves_reloads_and_dispatches_plots(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            patch.object(
                PartialWidthHistogramPlot,
                "plot",
                autospec=True,
            ) as partial_plot,
            patch.object(
                TotalWidthHistogramPlot,
                "plot",
                autospec=True,
            ) as total_plot,
        ):
            simulation = run_partial_widths_statistics_simulation(
                compound=build_compound(seed=311),
                width_indices=((0, 0), (0,)),
                realizs=1,
                directory=temporary_directory,
            )

            self.assertEqual(simulation.execution_state, ExecutionState.COMPLETE)
            partial_plot.assert_called_once()
            total_plot.assert_called_once()
            completion_time = simulation.manifest.execution["execution_time"]
            destination_directory = (
                Path(temporary_directory) / simulation.to_path / str(completion_time)
            )
            self.assertTrue((destination_directory / "manifest.json").is_file())
            self.assertEqual(
                len(tuple(destination_directory.rglob("*_data.npz"))),
                len(tuple(simulation)),
            )


if __name__ == "__main__":
    unittest.main()
