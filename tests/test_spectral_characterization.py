# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportUnusedCallResult=false, reportPrivateUsage=false, reportImplicitStringConcatenation=false, reportMissingParameterType=false, reportUnnecessaryIsInstance=false, reportImplicitOverride=false, reportExplicitAny=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false

import tempfile
import unittest
from copy import deepcopy
from unittest.mock import patch

import numpy as np

from rmtpy.ensembles import GOE
from rmtpy.simulations.base_simulation import ExecutionState as ExecutionState
from rmtpy.simulations.histogram import Histogram
from rmtpy.simulations.spectral_statistics import (
    SpectralStatisticsRequest,
    SpectralStatisticsSimulation,
    load_spectral_statistics_result,
    plot_spectral_statistics_result,
    save_spectral_statistics_result,
)
from rmtpy.simulations.spectral_statistics.spectral_histogram import (
    SpectralHistogramPlot,
)
from rmtpy.simulations.spectral_statistics.spectral_statistics_results import (
    SpectralStatisticsResult,
)
from rmtpy.simulations.statistics import nearest_neighbor_spacings
from rmtpy.simulations.unfolding import TruncatedPolynomialCDFFactory, unfold_values


def histogram_counts(samples: list[np.ndarray], bins: np.ndarray) -> np.ndarray:
    counts = np.zeros(len(bins) - 1, dtype=np.int64)
    for sample in samples:
        indices = np.searchsorted(bins, sample, side="right") - 1
        valid = (indices >= 0) & (indices < len(counts))
        np.add.at(counts, indices[valid], 1)
    return counts


def form_factor_moments(
    samples: list[np.ndarray],
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    first = np.zeros(len(times), dtype=np.complex128)
    second = np.zeros(len(times), dtype=np.float64)
    for sample in samples:
        contribution = np.mean(np.exp(-1j * np.outer(sample, times)), axis=0)
        first += contribution
        second += np.abs(contribution) ** 2
    return first, second


class SpectralStatisticsTests(unittest.TestCase):
    def test_default_result_schema_at_zero_one_and_all_degrees(self) -> None:
        expected = {
            0: (
                (),
                (
                    "spectral_histogram_data",
                    "spacings_histogram_data",
                    "spectral_form_factors_data",
                    "spectral_histogram_weight_unfolded_data",
                    "spacings_histogram_weight_unfolded_data",
                    "spectral_form_factors_weight_unfolded_data",
                ),
            ),
            1: (
                (1,),
                (
                    "spectral_coeff_1_histogram_data",
                    "spectral_histogram_data",
                    "spacings_histogram_data",
                    "spectral_form_factors_data",
                    "spectral_histogram_weight_unfolded_data",
                    "spacings_histogram_weight_unfolded_data",
                    "spectral_form_factors_weight_unfolded_data",
                    "spectral_histogram_average_unfolded_degree_1_data",
                    "spacings_histogram_average_unfolded_degree_1_data",
                    "spectral_form_factors_average_unfolded_degree_1_data",
                    "spectral_histogram_variate_unfolded_degree_1_data",
                    "spacings_histogram_variate_unfolded_degree_1_data",
                    "spectral_form_factors_variate_unfolded_degree_1_data",
                ),
            ),
            4: (
                (1, 2, 3, 4),
                (
                    "spectral_coeff_1_histogram_data",
                    "spectral_coeff_2_histogram_data",
                    "spectral_coeff_3_histogram_data",
                    "spectral_coeff_4_histogram_data",
                    "spectral_histogram_data",
                    "spacings_histogram_data",
                    "spectral_form_factors_data",
                    "spectral_histogram_weight_unfolded_data",
                    "spacings_histogram_weight_unfolded_data",
                    "spectral_form_factors_weight_unfolded_data",
                    "spectral_histogram_average_unfolded_degree_1_data",
                    "spacings_histogram_average_unfolded_degree_1_data",
                    "spectral_form_factors_average_unfolded_degree_1_data",
                    "spectral_histogram_average_unfolded_degree_2_data",
                    "spacings_histogram_average_unfolded_degree_2_data",
                    "spectral_form_factors_average_unfolded_degree_2_data",
                    "spectral_histogram_average_unfolded_degree_3_data",
                    "spacings_histogram_average_unfolded_degree_3_data",
                    "spectral_form_factors_average_unfolded_degree_3_data",
                    "spectral_histogram_average_unfolded_degree_4_data",
                    "spacings_histogram_average_unfolded_degree_4_data",
                    "spectral_form_factors_average_unfolded_degree_4_data",
                    "spectral_histogram_variate_unfolded_degree_1_data",
                    "spacings_histogram_variate_unfolded_degree_1_data",
                    "spectral_form_factors_variate_unfolded_degree_1_data",
                    "spectral_histogram_variate_unfolded_degree_2_data",
                    "spacings_histogram_variate_unfolded_degree_2_data",
                    "spectral_form_factors_variate_unfolded_degree_2_data",
                    "spectral_histogram_variate_unfolded_degree_3_data",
                    "spacings_histogram_variate_unfolded_degree_3_data",
                    "spectral_form_factors_variate_unfolded_degree_3_data",
                    "spectral_histogram_variate_unfolded_degree_4_data",
                    "spacings_histogram_variate_unfolded_degree_4_data",
                    "spectral_form_factors_variate_unfolded_degree_4_data",
                ),
            ),
        }

        for max_degree, (degrees, names) in expected.items():
            with self.subTest(max_degree=max_degree):
                simulation = SpectralStatisticsSimulation(
                    ensemble=GOE(
                        num_majoranas=4,
                        max_spectral_polynomial_degree=max_degree,
                        seed=17,
                    ),
                    realizs=1,
                )
                result = simulation.execute()

                self.assertFalse(hasattr(simulation, "outputs"))
                self.assertEqual(simulation.truncated_degrees, degrees)
                self.assertEqual(
                    tuple(data.file_name for data in result.iterate_data()),
                    names,
                )
                self.assertEqual(
                    tuple(data.metadata["degree"] for data in result.coefficients),
                    tuple(range(1, max_degree + 1)),
                )
                self.assertEqual(
                    tuple(item.degree for item in result.average_by_degree),
                    degrees,
                )
                self.assertEqual(
                    tuple(item.degree for item in result.variate_by_degree),
                    degrees,
                )

    def test_seeded_sampling_order_and_accumulators(self) -> None:
        simulation = SpectralStatisticsSimulation(
            ensemble=GOE(
                num_majoranas=4,
                max_spectral_polynomial_degree=2,
                seed=314159,
            ),
            realizs=2,
        )
        control = GOE(
            num_majoranas=4,
            max_spectral_polynomial_degree=2,
            seed=314159,
        )
        initial_rng_state = deepcopy(simulation.ensemble.rng_state)

        self.assertEqual(initial_rng_state, control.rng_state)
        self.assertFalse(simulation.ensemble.spectral_density.has_average_coeffs)

        control_factory = TruncatedPolynomialCDFFactory(
            density=control.spectral_density,
            degrees=(1, 2),
            density_name="spectral",
        )
        average_cdfs = control_factory.average_interpolators()
        samples = list(control.eigvals_stream(realizs=2))
        result = simulation.execute()

        self.assertEqual(simulation.ensemble.rng_state, control.rng_state)
        self.assertNotEqual(simulation.ensemble.rng_state, initial_rng_state)
        self.assertTrue(simulation.ensemble.spectral_density.has_average_coeffs)

        raw = result.raw
        spacings = [nearest_neighbor_spacings(sample, degeneracy=1) for sample in samples]
        np.testing.assert_array_equal(
            raw.levels.counts,
            histogram_counts(samples, raw.levels.bins),
        )
        np.testing.assert_array_equal(
            raw.spacings.counts,
            histogram_counts(spacings, raw.spacings.bins),
        )
        expected_first, expected_second = form_factor_moments(
            samples,
            raw.form_factors.times,
        )
        np.testing.assert_allclose(raw.form_factors.first_moment, expected_first)
        np.testing.assert_allclose(raw.form_factors.second_moment, expected_second)

        coefficient_samples = [
            control.spectral_density.compute_variate_coeffs(sample)[1:]
            for sample in samples
        ]
        for index, histogram in enumerate(result.coefficients):
            values = [
                coefficients[index : index + 1] for coefficients in coefficient_samples
            ]
            np.testing.assert_array_equal(
                histogram.counts,
                histogram_counts(values, histogram.bins),
            )

        weight_samples = [
            unfold_values(
                sample,
                cdf=control.spectral_density.weight_cdf,
                dimension=control.dimension,
            )
            for sample in samples
        ]
        np.testing.assert_array_equal(
            result.weight.levels.counts,
            histogram_counts(weight_samples, result.weight.levels.bins),
        )

        for average_cdf, item in zip(
            average_cdfs,
            result.average_by_degree,
            strict=True,
        ):
            average_samples = [
                unfold_values(sample, cdf=average_cdf, dimension=control.dimension)
                for sample in samples
            ]
            np.testing.assert_array_equal(
                item.statistics.levels.counts,
                histogram_counts(average_samples, item.statistics.levels.bins),
            )

        variate_samples: list[list[np.ndarray]] = [[] for _ in result.variate_by_degree]
        for sample in samples:
            variate_cdfs = control_factory.interpolators_from_coeffs(
                control.spectral_density.compute_variate_coeffs(sample)
            )
            for position, variate_cdf in enumerate(variate_cdfs):
                variate_samples[position].append(
                    unfold_values(
                        sample,
                        cdf=variate_cdf,
                        dimension=control.dimension,
                    )
                )

        for item, unfolded_samples in zip(
            result.variate_by_degree,
            variate_samples,
            strict=True,
        ):
            np.testing.assert_array_equal(
                item.statistics.levels.counts,
                histogram_counts(unfolded_samples, item.statistics.levels.bins),
            )

        for data in result.iterate_data():
            if isinstance(data, Histogram) and np.sum(data.counts):
                np.testing.assert_allclose(
                    data.histogram,
                    data.counts / (np.sum(data.counts) * np.diff(data.bins)),
                )

    def test_result_context_and_one_shot_lifecycle(self) -> None:
        simulation = SpectralStatisticsSimulation(
            ensemble=GOE(
                num_majoranas=4,
                max_spectral_polynomial_degree=2,
                seed=31,
            ),
            realizs=1,
        )
        initial_rng_state = deepcopy(simulation.ensemble.rng_state)
        result = simulation.execute()

        self.assertIsInstance(result, SpectralStatisticsResult)
        self.assertIs(simulation.result, result)
        self.assertEqual(result.context.rng["initial_state"], initial_rng_state)
        self.assertEqual(result.context.rng["final_state"], simulation.ensemble.rng_state)
        self.assertNotIn("request", result.context.simulation_config["parameters"])
        self.assertEqual(
            result.context.output_request,
            {
                "quantities": [
                    "coefficients",
                    "levels",
                    "spacings",
                    "form_factors",
                ],
                "unfolding_modes": ["raw", "weight", "average", "variate"],
                "degrees": [1, 2],
                "max_degree": 2,
            },
        )
        self.assertEqual(simulation.execution_state, ExecutionState.COMPLETE)
        with self.assertRaisesRegex(RuntimeError, "only once"):
            simulation.execute()

    def test_failed_execution_is_terminal(self) -> None:
        simulation = SpectralStatisticsSimulation(
            ensemble=GOE(num_majoranas=4, seed=32),
            realizs=1,
        )
        with (
            patch.object(
                SpectralStatisticsSimulation,
                "_realize_spectral_statistics",
                side_effect=RuntimeError("expected failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "expected failure"),
        ):
            simulation.execute()

        self.assertEqual(simulation.execution_state, ExecutionState.FAILED)
        with self.assertRaisesRegex(RuntimeError, "only once"):
            simulation.execute()
        with self.assertRaisesRegex(RuntimeError, "not completed"):
            _ = simulation.result

    def test_result_persistence_and_selective_plotting(self) -> None:
        simulation = SpectralStatisticsSimulation(
            ensemble=GOE(
                num_majoranas=4,
                max_spectral_polynomial_degree=0,
                seed=311,
            ),
            realizs=1,
        )
        result = simulation.execute()
        completed_rng_state = deepcopy(simulation.ensemble.rng_state)

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = save_spectral_statistics_result(result, out_dir=tmp_dir)
            restored = load_spectral_statistics_result(run_dir)
            np.testing.assert_array_equal(
                restored.raw.levels.counts,
                result.raw.levels.counts,
            )
            np.testing.assert_allclose(
                restored.raw.form_factors.form_factor,
                result.raw.form_factors.form_factor,
            )

            with patch.object(
                SpectralHistogramPlot,
                "plot",
                autospec=True,
            ) as render:
                plot_spectral_statistics_result(
                    result,
                    out_dir=tmp_dir,
                    views="spectral_histogram",
                )

            render.assert_called_once()
            self.assertIs(render.call_args.args[0].data, result.raw.levels)

        self.assertEqual(simulation.ensemble.rng_state, completed_rng_state)

    def test_execute_is_data_only(self) -> None:
        simulation = SpectralStatisticsSimulation(
            ensemble=GOE(num_majoranas=4, seed=33),
            realizs=1,
        )
        with patch("rmtpy.simulations.persistence.save_run") as save_result:
            result = simulation.execute()

        self.assertIs(result, simulation.result)
        save_result.assert_not_called()

    def test_selective_raw_levels_omit_unrequested_work(self) -> None:
        simulation = SpectralStatisticsSimulation(
            ensemble=GOE(
                num_majoranas=4,
                max_spectral_polynomial_degree=2,
                seed=34,
            ),
            realizs=1,
            request=SpectralStatisticsRequest(
                quantities=("levels",),
                unfolding_modes=("raw",),
            ),
        )
        density = simulation.ensemble.spectral_density

        with (
            patch.object(
                type(density),
                "compute_variate_coeffs",
                side_effect=AssertionError("coefficients were unrequested"),
            ),
            patch.object(
                type(density),
                "_compute_average_coeffs",
                side_effect=AssertionError("average calibration was unrequested"),
            ),
            patch(
                "rmtpy.simulations.spectral_statistics."
                "spectral_statistics_simulation.unfold_values",
                side_effect=AssertionError("unfolding was unrequested"),
            ),
        ):
            result = simulation.execute()

        self.assertIsNotNone(result.raw.levels)
        self.assertIsNone(result.raw.spacings)
        self.assertIsNone(result.raw.form_factors)
        self.assertEqual(result.coefficients, ())
        self.assertIsNone(result.weight)
        self.assertEqual(result.average_by_degree, ())
        self.assertEqual(result.variate_by_degree, ())
        self.assertEqual(tuple(result.iterate_data()), (result.raw.levels,))

    def test_variate_levels_compute_internal_coefficients(self) -> None:
        simulation = SpectralStatisticsSimulation(
            ensemble=GOE(
                num_majoranas=4,
                max_spectral_polynomial_degree=2,
                seed=35,
            ),
            realizs=1,
            request=SpectralStatisticsRequest(
                quantities=("levels",),
                unfolding_modes=("variate",),
                degrees=(2,),
            ),
        )
        result = simulation.execute()

        self.assertEqual(result.coefficients, ())
        self.assertIsNone(result.raw)
        self.assertEqual(len(result.variate_by_degree), 1)
        self.assertEqual(result.variate_by_degree[0].degree, 2)
        self.assertEqual(len(tuple(result.iterate_data())), 1)

    def test_selective_request_changes_rng_trajectory(self) -> None:
        complete = SpectralStatisticsSimulation(
            ensemble=GOE(
                num_majoranas=4,
                max_spectral_polynomial_degree=2,
                seed=36,
            ),
            realizs=1,
        )
        selective = SpectralStatisticsSimulation(
            ensemble=GOE(
                num_majoranas=4,
                max_spectral_polynomial_degree=2,
                seed=36,
            ),
            realizs=1,
            request=SpectralStatisticsRequest(
                quantities=("levels",),
                unfolding_modes=("raw",),
            ),
        )

        complete_result = complete.execute()
        selective_result = selective.execute()

        self.assertNotEqual(
            complete_result.context.rng["final_state"],
            selective_result.context.rng["final_state"],
        )
        self.assertFalse(
            np.array_equal(
                complete_result.raw.levels.counts,
                selective_result.raw.levels.counts,
            )
        )

    def test_request_validation_uses_only_canonical_terms(self) -> None:
        request = SpectralStatisticsRequest(
            quantities=("form_factors", "levels"),
            unfolding_modes=("variate", "weight"),
            degrees=(2,),
        )
        self.assertEqual(request.quantities, ("levels", "form_factors"))
        self.assertEqual(request.unfolding_modes, ("weight", "variate"))

        for unsupported_name in ("wgt", "avg", "var"):
            with (
                self.subTest(unsupported_name=unsupported_name),
                self.assertRaises(ValueError),
            ):
                SpectralStatisticsRequest(unfolding_modes=(unsupported_name,))

        with self.assertRaises(ValueError):
            SpectralStatisticsRequest(quantities=("levels", "levels"))
        with self.assertRaises(ValueError):
            SpectralStatisticsSimulation(
                ensemble=GOE(
                    num_majoranas=4,
                    max_spectral_polynomial_degree=2,
                ),
                realizs=1,
                request=SpectralStatisticsRequest(degrees=(4,)),
            )

    def test_realization_count_degeneracy_and_half_open_support(self) -> None:
        for value in (0, -1):
            with self.subTest(realizs=value), self.assertRaises(ValueError):
                SpectralStatisticsSimulation(
                    ensemble=GOE(num_majoranas=4),
                    realizs=value,
                )

        np.testing.assert_array_equal(
            nearest_neighbor_spacings(
                np.array([0.0, 0.0, 1.0, 1.0, 3.0, 3.0]),
                degeneracy=2,
            ),
            np.array([1.0, 1.0, 2.0, 2.0]),
        )

        histogram = Histogram(file_name="half_open", support=(0.0, 1.0), num_bins=2)
        histogram.add_histogram_contribution(np.array([-0.1, 0.0, 0.5, 1.0]))
        np.testing.assert_array_equal(histogram.counts, np.array([1, 1]))


if __name__ == "__main__":
    unittest.main()
