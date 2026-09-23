# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportUnusedCallResult=false, reportPrivateUsage=false, reportImplicitStringConcatenation=false, reportMissingParameterType=false, reportUnnecessaryIsInstance=false, reportImplicitOverride=false, reportExplicitAny=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false

import tempfile
import unittest
from copy import deepcopy
from unittest.mock import patch

import numpy as np

from rmtpy.compounds import CompoundEnsemble
from rmtpy.ensembles import GOE
from rmtpy.simulations.base_simulation import SimulationExecutionState as ExecutionState
from rmtpy.simulations.histogram2D import Histogram2D
from rmtpy.simulations.resonance_statistics import (
    ResonanceStatisticsRequest,
    ResonanceStatisticsResult,
    ResonanceStatisticsSimulation,
    load_resonance_statistics_result,
    plot_resonance_statistics_result,
    save_resonance_statistics_result,
)
from rmtpy.simulations.resonance_statistics.resonance_histogram import (
    ResonanceHistogramPlot,
)
from rmtpy.simulations.unfolding import (
    TruncatedPolynomialCdfFactory,
    unfold_values,
    unfold_widths,
)


def build_compound(*, max_degree: int = 0, seed: int = 123) -> CompoundEnsemble:
    return CompoundEnsemble(
        ensemble=GOE(
            num_majoranas=4,
            max_spectral_polynomial_degree=max_degree,
            seed=seed,
        ),
        couplings=np.array([0.75, 1.25]),
    )


def histogram_counts(samples: list[np.ndarray], bins: np.ndarray) -> np.ndarray:
    counts = np.zeros(len(bins) - 1, dtype=np.int64)
    for sample in samples:
        indices = np.searchsorted(bins, sample, side="right") - 1
        valid = (indices >= 0) & (indices < len(counts))
        np.add.at(counts, indices[valid], 1)
    return counts


def histogram2d_counts(
    x_values: np.ndarray,
    y_values: np.ndarray,
    histogram: Histogram2D,
) -> np.ndarray:
    counts = np.zeros_like(histogram.counts)
    x_indices = np.searchsorted(histogram.x_bins, x_values, side="right") - 1
    y_indices = np.searchsorted(histogram.y_bins, y_values, side="right") - 1
    valid = (
        (x_indices >= 0)
        & (x_indices < counts.shape[0])
        & (y_indices >= 0)
        & (y_indices < counts.shape[1])
    )
    np.add.at(counts, (x_indices[valid], y_indices[valid]), 1)
    return counts


class ResonanceStatisticsTests(unittest.TestCase):
    def test_poles_become_centers_widths_and_requested_raw_quantities(self) -> None:
        simulation = ResonanceStatisticsSimulation(
            compound=build_compound(),
            realizs=1,
            request=ResonanceStatisticsRequest(
                quantities=(
                    "resonances",
                    "widths",
                    "spacings",
                    "complex_energies",
                    "form_factors",
                ),
                unfolding_modes=("raw", "weight"),
            ),
        )
        poles = np.array([-0.5 - 0.1j, 0.25 - 0.4j])
        centers = np.real(poles)
        widths = -2.0 * np.imag(poles)
        radius = simulation.compound.ensemble.spectral_radius

        with patch.object(
            CompoundEnsemble,
            "resonances_stream",
            return_value=iter((poles,)),
        ):
            result = simulation.execute()

        raw = result.raw
        np.testing.assert_array_equal(
            raw.resonances.counts,
            histogram_counts([centers], raw.resonances.bins),
        )
        np.testing.assert_array_equal(
            raw.widths.counts,
            histogram_counts([widths / radius], raw.widths.bins),
        )
        np.testing.assert_array_equal(
            raw.spacings.counts,
            histogram_counts([np.array([0.75])], raw.spacings.bins),
        )
        expected_raw_2d = histogram2d_counts(
            centers / radius,
            widths / radius,
            raw.complex_energies,
        )
        np.testing.assert_array_equal(raw.complex_energies.counts, expected_raw_2d)
        expected_moment = np.mean(
            np.exp(-1j * np.outer(centers, raw.form_factors.times)),
            axis=0,
        )
        np.testing.assert_allclose(raw.form_factors.first_moment, expected_moment)

        weight_centers = unfold_values(
            centers,
            cdf=simulation.compound.resonance_density.weight_cdf,
            dimension=simulation.compound.ensemble.dimension,
        )
        weight_widths = unfold_widths(
            widths,
            centers=centers,
            cdf=simulation.compound.resonance_density.weight_cdf,
            dimension=simulation.compound.ensemble.dimension,
        )
        weight = result.weight
        np.testing.assert_array_equal(
            weight.resonances.counts,
            histogram_counts([weight_centers], weight.resonances.bins),
        )
        np.testing.assert_array_equal(
            weight.widths.counts,
            histogram_counts([weight_widths], weight.widths.bins),
        )
        expected_mixed_2d = histogram2d_counts(
            centers / radius,
            weight_widths,
            weight.complex_energies,
        )
        np.testing.assert_array_equal(
            weight.complex_energies.counts,
            expected_mixed_2d,
        )
        self.assertEqual(float(np.sum(weight.complex_energies.histogram)), 1.0)

    def test_schema_degree_order_and_canonical_metadata(self) -> None:
        expected = {0: ((), 10), 2: ((1, 2), 32)}
        for max_degree, (degrees, data_count) in expected.items():
            with self.subTest(max_degree=max_degree):
                simulation = ResonanceStatisticsSimulation(
                    compound=build_compound(max_degree=max_degree, seed=40 + max_degree),
                    realizs=1,
                )
                result = simulation.execute()
                self.assertIsInstance(result, ResonanceStatisticsResult)
                self.assertFalse(hasattr(simulation, "outputs"))
                self.assertEqual(simulation.truncated_degrees, degrees)
                self.assertEqual(len(tuple(result.iterate_data())), data_count)
                self.assertEqual(
                    tuple(data.metadata["degree"] for data in result.coefficients),
                    tuple(range(1, max_degree + 1)),
                )
                self.assertEqual(result.raw.widths.metadata["unfolding"], "raw")
                self.assertEqual(result.weight.widths.metadata["unfolding"], "weight")
                self.assertEqual(
                    tuple(item.degree for item in result.average_by_degree),
                    degrees,
                )
                self.assertEqual(
                    tuple(item.degree for item in result.variate_by_degree),
                    degrees,
                )

    def test_selective_width_request_omits_other_work_and_grids(self) -> None:
        simulation = ResonanceStatisticsSimulation(
            compound=build_compound(max_degree=2),
            realizs=1,
            request=ResonanceStatisticsRequest(
                quantities=("widths",),
                unfolding_modes=("raw",),
            ),
        )
        density = simulation.compound.resonance_density
        poles = np.array([-0.25 - 0.5j, 0.25 - 0.25j])
        with (
            patch.object(
                CompoundEnsemble,
                "resonances_stream",
                return_value=iter((poles,)),
            ),
            patch.object(
                type(density),
                "compute_variate_coeffs",
                side_effect=AssertionError("coefficients were unrequested"),
            ),
            patch.object(
                type(density),
                "_compute_average_coeffs",
                side_effect=AssertionError("calibration was unrequested"),
            ),
            patch(
                "rmtpy.simulations.resonance_statistics."
                "resonance_statistics_simulation.unfold_values",
                side_effect=AssertionError("center unfolding was unrequested"),
            ),
            patch(
                "rmtpy.simulations.resonance_statistics."
                "resonance_statistics_simulation.unfold_widths",
                side_effect=AssertionError("width unfolding was unrequested"),
            ),
        ):
            result = simulation.execute()

        self.assertEqual(result.coefficients, ())
        self.assertIsNotNone(result.raw.widths)
        self.assertIsNone(result.raw.resonances)
        self.assertIsNone(result.raw.complex_energies)
        self.assertIsNone(result.weight)
        self.assertEqual(len(tuple(result.iterate_data())), 1)

    def test_variate_request_computes_internal_coefficients_without_calibration(
        self,
    ) -> None:
        simulation = ResonanceStatisticsSimulation(
            compound=build_compound(max_degree=2, seed=51),
            realizs=1,
            request=ResonanceStatisticsRequest(
                quantities=("resonances",),
                unfolding_modes=("variate",),
                degrees=(2,),
            ),
        )
        density = simulation.compound.resonance_density
        with patch.object(
            type(density),
            "_compute_average_coeffs",
            side_effect=AssertionError("average calibration was unrequested"),
        ):
            result = simulation.execute()

        self.assertEqual(result.coefficients, ())
        self.assertEqual(len(result.variate_by_degree), 1)
        self.assertEqual(len(tuple(result.iterate_data())), 1)

    def test_average_calibration_timing_and_seeded_rng_match(self) -> None:
        simulation = ResonanceStatisticsSimulation(
            compound=build_compound(max_degree=2, seed=314159),
            realizs=2,
        )
        control = build_compound(max_degree=2, seed=314159)
        initial_rng_state = deepcopy(simulation.compound.rng_state)
        factory = TruncatedPolynomialCdfFactory(
            density=control.resonance_density,
            degrees=(1, 2),
            density_name="resonance",
        )
        stream = control.resonances_stream(realizs=2)

        first_poles = next(stream)
        first_centers = np.real(first_poles)
        first_coefficients = control.resonance_density.compute_variate_coeffs(
            first_centers
        )
        average_cdf = factory.average_interpolators()[0]
        first_variate_cdf = factory.interpolators_from_coeffs(first_coefficients)[0]

        second_poles = next(stream)
        second_centers = np.real(second_poles)
        second_coefficients = control.resonance_density.compute_variate_coeffs(
            second_centers
        )
        second_variate_cdf = factory.interpolators_from_coeffs(second_coefficients)[0]

        result = simulation.execute()

        self.assertEqual(simulation.compound.rng_state, control.rng_state)
        self.assertEqual(result.context.rng["initial_state"], initial_rng_state)
        self.assertEqual(result.context.rng["final_state"], control.rng_state)
        for degree_index, histogram in enumerate(result.coefficients, start=1):
            values = [
                first_coefficients[degree_index : degree_index + 1],
                second_coefficients[degree_index : degree_index + 1],
            ]
            np.testing.assert_array_equal(
                histogram.counts,
                histogram_counts(values, histogram.bins),
            )

        average_samples = [
            unfold_values(
                centers,
                cdf=average_cdf,
                dimension=control.ensemble.dimension,
            )
            for centers in (first_centers, second_centers)
        ]
        average_histogram = result.average_by_degree[0].statistics.resonances
        np.testing.assert_array_equal(
            average_histogram.counts,
            histogram_counts(average_samples, average_histogram.bins),
        )
        variate_samples = [
            unfold_values(
                centers,
                cdf=cdf,
                dimension=control.ensemble.dimension,
            )
            for centers, cdf in (
                (first_centers, first_variate_cdf),
                (second_centers, second_variate_cdf),
            )
        ]
        variate_histogram = result.variate_by_degree[0].statistics.resonances
        np.testing.assert_array_equal(
            variate_histogram.counts,
            histogram_counts(variate_samples, variate_histogram.bins),
        )

    def test_request_validation_and_failed_lifecycle(self) -> None:
        request = ResonanceStatisticsRequest(
            quantities=("form_factors", "widths"),
            unfolding_modes=("variate", "weight"),
            degrees=(2,),
        )
        self.assertEqual(request.quantities, ("widths", "form_factors"))
        self.assertEqual(request.unfolding_modes, ("weight", "variate"))
        for legacy_name in ("wgt", "avg", "var"):
            with self.subTest(legacy_name=legacy_name), self.assertRaises(ValueError):
                ResonanceStatisticsRequest(unfolding_modes=(legacy_name,))

        simulation = ResonanceStatisticsSimulation(
            compound=build_compound(),
            realizs=1,
        )
        with (
            patch.object(
                ResonanceStatisticsSimulation,
                "_realize_resonance_statistics",
                side_effect=RuntimeError("expected failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "expected failure"),
        ):
            simulation.execute()
        self.assertEqual(simulation.execution_state, ExecutionState.FAILED)
        with self.assertRaisesRegex(RuntimeError, "only once"):
            simulation.execute()

    def test_persistence_plotting_and_run_consumers(self) -> None:
        simulation = ResonanceStatisticsSimulation(
            compound=build_compound(seed=88),
            realizs=1,
            request=ResonanceStatisticsRequest(
                quantities=("resonances",),
                unfolding_modes=("raw",),
            ),
        )
        result = simulation.execute()
        completed_rng_state = deepcopy(simulation.compound.rng_state)
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = save_resonance_statistics_result(result, out_dir=tmp_dir)
            restored = load_resonance_statistics_result(run_dir)
            np.testing.assert_array_equal(
                restored.raw.resonances.counts,
                result.raw.resonances.counts,
            )

            with patch.object(ResonanceHistogramPlot, "plot", autospec=True) as render:
                plot_resonance_statistics_result(
                    result,
                    out_dir=tmp_dir,
                    views="resonance_histogram",
                )
            render.assert_called_once()
            self.assertIs(render.call_args.args[0].data, result.raw.resonances)
        self.assertEqual(simulation.compound.rng_state, completed_rng_state)

        data_only = ResonanceStatisticsSimulation(
            compound=build_compound(seed=89),
            realizs=1,
            request=ResonanceStatisticsRequest(
                quantities=("widths",),
                unfolding_modes=("raw",),
            ),
        )
        with patch("rmtpy.simulations.persistence.save_run") as save_result:
            run_result = data_only.execute()
        self.assertIs(run_result, data_only.result)
        save_result.assert_not_called()


if __name__ == "__main__":
    unittest.main()
