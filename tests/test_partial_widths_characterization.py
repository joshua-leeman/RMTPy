# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportUnusedCallResult=false, reportPrivateUsage=false, reportImplicitStringConcatenation=false, reportMissingParameterType=false, reportUnnecessaryIsInstance=false, reportImplicitOverride=false, reportExplicitAny=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false

import tempfile
import unittest
from copy import deepcopy
from unittest.mock import patch

import numpy as np

from rmtpy.compounds import CompoundEnsemble
from rmtpy.ensembles import GOE
from rmtpy.simulations.base_simulation import ExecutionState as ExecutionState
from rmtpy.simulations.partial_widths_statistics import (
    PartialWidthsStatisticsResult,
    PartialWidthsStatisticsSimulation,
    PartialWidthStatistic,
    TotalWidthStatistic,
    load_partial_widths_statistics_result,
    plot_partial_widths_statistics_result,
    save_partial_widths_statistics_result,
)
from rmtpy.simulations.partial_widths_statistics.partial_width_histogram import (
    PartialWidthHistogramPlot,
)
from rmtpy.simulations.partial_widths_statistics.partial_widths_statistics_simulation import (
    build_width_statistics,
)


def build_compound(
    *, num_free_complex_fermions: int = 1, seed: int = 123
) -> CompoundEnsemble:
    return CompoundEnsemble(
        ensemble=GOE(num_majoranas=4, seed=seed),
        num_free_complex_fermions=num_free_complex_fermions,
    )


def histogram_counts(samples: list[float], bins: np.ndarray) -> np.ndarray:
    counts = np.zeros(len(bins) - 1, dtype=np.int64)
    indices = np.searchsorted(bins, samples, side="right") - 1
    valid = (indices >= 0) & (indices < len(counts))
    np.add.at(counts, indices[valid], 1)
    return counts


class PartialWidthsStatisticsTests(unittest.TestCase):
    def test_partial_and_total_transformations_preserve_request_order(self) -> None:
        simulation = PartialWidthsStatisticsSimulation(
            compound=build_compound(),
            realizs=2,
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
            result = simulation.execute()

        self.assertIsInstance(result, PartialWidthsStatisticsResult)
        self.assertEqual(
            tuple(statistic.index for statistic in result.statistics),
            simulation.width_indices,
        )
        self.assertIsInstance(result.statistics[0], PartialWidthStatistic)
        self.assertIsInstance(result.statistics[3], TotalWidthStatistic)

        raw_values = ((2.0, 4.0), (6.0, 12.0), (8.0, 16.0), (6.0, 12.0), (14.0, 28.0))
        expected_means = (3.0, 9.0, 12.0, 9.0, 21.0)
        for statistic, values, expected_mean in zip(
            result.statistics,
            raw_values,
            expected_means,
            strict=True,
        ):
            histogram = statistic.histogram
            unscaled_bins = histogram.bins * expected_mean
            expected_counts = histogram_counts(list(values), unscaled_bins)
            np.testing.assert_array_equal(histogram.counts, expected_counts)
            self.assertEqual(histogram.realizs, 2)
            self.assertEqual(histogram.metadata["average_width"], expected_mean)
            np.testing.assert_allclose(
                histogram.histogram,
                expected_counts / (np.sum(expected_counts) * np.diff(histogram.bins)),
            )

    def test_default_and_invalid_indices_follow_compound_shape(self) -> None:
        self.assertEqual(
            PartialWidthsStatisticsSimulation(
                compound=build_compound(), realizs=1
            ).width_indices,
            ((0, 0), (1, 0), (1, 1), (0,), (1,)),
        )
        self.assertEqual(
            PartialWidthsStatisticsSimulation(
                compound=build_compound(num_free_complex_fermions=0),
                realizs=1,
            ).width_indices,
            ((0, 0), (1, 0), (0,), (1,)),
        )

        compound = build_compound(num_free_complex_fermions=0)
        invalid_cases = ((), ((0, 0), (0, 0)), ((0, 0, 0),), ((2,),), ((0, 1),))
        for width_indices in invalid_cases:
            with self.subTest(width_indices=width_indices), self.assertRaises(ValueError):
                PartialWidthsStatisticsSimulation(
                    compound=compound,
                    realizs=1,
                    width_indices=width_indices,
                )

    def test_selective_request_allocates_only_selected_histograms(self) -> None:
        simulation = PartialWidthsStatisticsSimulation(
            compound=build_compound(),
            realizs=1,
            width_indices=((1,),),
        )
        self.assertFalse(hasattr(simulation, "outputs"))
        self.assertFalse(hasattr(simulation, "_buffers"))

        with patch.object(
            CompoundEnsemble,
            "partial_widths_stream",
            return_value=iter((np.array([[1.0, 2.0], [3.0, 4.0]]),)),
        ):
            result = simulation.execute()

        self.assertEqual(len(result.statistics), 1)
        self.assertIsInstance(result.statistics[0], TotalWidthStatistic)
        self.assertEqual(tuple(result.iterate_data()), (result.statistics[0].histogram,))

    def test_zero_mean_is_terminal_and_does_not_rescale_bins(self) -> None:
        statistics = build_width_statistics(((0, 0), (0,)))
        initial_bins = tuple(statistic.histogram.bins.copy() for statistic in statistics)
        simulation = PartialWidthsStatisticsSimulation(
            compound=build_compound(),
            realizs=1,
            width_indices=((0, 0), (0,)),
        )

        with (
            patch(
                "rmtpy.simulations.partial_widths_statistics."
                "partial_widths_statistics_simulation.build_width_statistics",
                return_value=statistics,
            ),
            patch.object(
                CompoundEnsemble,
                "partial_widths_stream",
                return_value=iter((np.zeros((2, 2)),)),
            ),
            self.assertRaisesRegex(ValueError, "positive and finite"),
        ):
            simulation.execute()

        self.assertEqual(simulation.execution_state, ExecutionState.FAILED)
        for statistic, bins in zip(statistics, initial_bins, strict=True):
            self.assertEqual(statistic.histogram.metadata["average_width"], 0.0)
            np.testing.assert_array_equal(statistic.histogram.bins, bins)
        with self.assertRaisesRegex(RuntimeError, "not completed"):
            _ = simulation.result
        with self.assertRaisesRegex(RuntimeError, "only once"):
            simulation.execute()

    def test_seeded_result_and_rng_match_direct_stream(self) -> None:
        simulation = PartialWidthsStatisticsSimulation(
            compound=CompoundEnsemble(ensemble=GOE(num_majoranas=4, seed=314159)),
            realizs=2,
            width_indices=((0, 0), (1,)),
        )
        control = CompoundEnsemble(ensemble=GOE(num_majoranas=4, seed=314159))
        initial_rng_state = deepcopy(simulation.compound.rng_state)
        samples = list(control.partial_widths_stream(realizs=2))

        result = simulation.execute()

        self.assertEqual(simulation.compound.rng_state, control.rng_state)
        self.assertEqual(result.context.rng["initial_state"], initial_rng_state)
        self.assertEqual(result.context.rng["final_state"], control.rng_state)
        expected_values = (
            [float(sample[0, 0]) for sample in samples],
            [float(np.sum(sample[1])) for sample in samples],
        )
        for statistic, values in zip(result.statistics, expected_values, strict=True):
            mean = sum(values) / simulation.realizs
            np.testing.assert_array_equal(
                statistic.histogram.counts,
                histogram_counts(values, statistic.histogram.bins * mean),
            )
            self.assertEqual(statistic.histogram.metadata["average_width"], mean)

    def test_persistence_plotting_and_run_side_effects_are_explicit(self) -> None:
        simulation = PartialWidthsStatisticsSimulation(
            compound=build_compound(seed=77),
            realizs=1,
            width_indices=((0, 0),),
        )
        result = simulation.execute()
        completed_rng_state = deepcopy(simulation.compound.rng_state)

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = save_partial_widths_statistics_result(result, out_dir=tmp_dir)
            data = result.statistics[0].histogram
            stem = data.file_name.removesuffix("_data")
            restored = load_partial_widths_statistics_result(run_dir)
            np.testing.assert_array_equal(
                restored.statistics[0].histogram.counts,
                data.counts,
            )

            with patch.object(PartialWidthHistogramPlot, "plot", autospec=True) as render:
                plot_partial_widths_statistics_result(
                    result,
                    out_dir=tmp_dir,
                    views=stem,
                )
            render.assert_called_once()
            self.assertIs(render.call_args.args[0].data, data)

        self.assertEqual(simulation.compound.rng_state, completed_rng_state)

        data_only = PartialWidthsStatisticsSimulation(
            compound=build_compound(seed=78),
            realizs=1,
            width_indices=((0, 0),),
        )
        with patch("rmtpy.simulations.persistence.save_run") as save_result:
            run_result = data_only.execute()
        self.assertIs(run_result, data_only.result)
        save_result.assert_not_called()


if __name__ == "__main__":
    unittest.main()
