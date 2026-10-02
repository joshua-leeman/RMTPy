import json
import tempfile
import unittest
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np

from rmtpy.compounds import CompoundEnsemble
from rmtpy.ensembles import GOE
from rmtpy.simulations.base_simulation import ExecutionState
from rmtpy.simulations.resonance_statistics import (
    ResonanceStatisticsSimulation,
    load_resonance_statistics_simulation,
    plot_resonance_statistics_simulation,
    run_resonance_statistics_simulation,
)
from rmtpy.simulations.resonance_statistics.complex_energy_histogram import (
    ComplexEnergyHistogram,
    ComplexEnergyHistogramPlot,
    UnfoldedComplexEnergyHistogramPlot,
)
from rmtpy.simulations.resonance_statistics.resonance_coefficients_histogram import (
    ResonanceCoefficientsHistogram,
    ResonanceCoefficientsHistogramPlot,
)
from rmtpy.simulations.resonance_statistics.resonance_form_factors import (
    FormFactorsData,
    ResonanceFormFactorsPlot,
    UnfoldedResonanceFormFactorsPlot,
)
from rmtpy.simulations.resonance_statistics.resonance_histogram import (
    ResonanceHistogram,
    ResonanceHistogramPlot,
    UnfoldedResonanceHistogramPlot,
)
from rmtpy.simulations.resonance_statistics.resonance_spacing_histogram import (
    ResonanceSpacingHistogram,
    ResonanceSpacingHistogramPlot,
    UnfoldedResonanceSpacingHistogramPlot,
)
from rmtpy.simulations.resonance_statistics.width_histogram import (
    UnfoldedWidthHistogramPlot,
    WidthHistogram,
    WidthHistogramPlot,
)
from rmtpy.simulations.unfolding import (
    TruncatedPolynomialCDFFactory,
    unfold_values,
    unfold_widths,
)

RESONANCE_PLOT_CLASSES = (
    ResonanceCoefficientsHistogramPlot,
    ResonanceHistogramPlot,
    UnfoldedResonanceHistogramPlot,
    WidthHistogramPlot,
    UnfoldedWidthHistogramPlot,
    ResonanceSpacingHistogramPlot,
    UnfoldedResonanceSpacingHistogramPlot,
    UnfoldedComplexEnergyHistogramPlot,
    ComplexEnergyHistogramPlot,
    ResonanceFormFactorsPlot,
    UnfoldedResonanceFormFactorsPlot,
)

type ResonanceHistogramData = (
    ResonanceCoefficientsHistogram
    | ResonanceHistogram
    | WidthHistogram
    | ResonanceSpacingHistogram
    | ComplexEnergyHistogram
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


def histogram_counts(
    samples: tuple[np.ndarray[tuple[int], np.dtype[np.floating]], ...],
    *,
    bins: np.ndarray[tuple[int], np.dtype[np.floating]],
) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
    counts = np.zeros(len(bins) - 1, dtype=np.int64)
    for sample in samples:
        indices = np.searchsorted(bins, sample, side="right") - 1
        valid = (indices >= 0) & (indices < len(counts))
        np.add.at(counts, indices[valid], 1)

    return counts


def histogram2d_counts(
    x_values: np.ndarray[tuple[int], np.dtype[np.floating]],
    y_values: np.ndarray[tuple[int], np.dtype[np.floating]],
    *,
    histogram: ComplexEnergyHistogram,
) -> np.ndarray[tuple[int, int], np.dtype[np.int64]]:
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
    def test_buffer_schema_filenames_metadata_and_iteration_order(self) -> None:
        expected_counts = {0: 10, 2: 32}
        for max_degree, expected_count in expected_counts.items():
            with self.subTest(max_degree=max_degree):
                simulation = ResonanceStatisticsSimulation(
                    compound=build_compound(max_degree=max_degree),
                    realizs=1,
                )

                coefficient_buffers = tuple(simulation.coefficient_buffers)
                averaged_buffers = tuple(simulation.ave_unfolded_buffers)
                variate_buffers = tuple(simulation.var_unfolded_buffers)

                self.assertEqual(len(tuple(simulation)), expected_count)
                self.assertEqual(
                    tuple(type(buffer) for buffer in simulation.raw_buffers),
                    (
                        ResonanceHistogram,
                        WidthHistogram,
                        ResonanceSpacingHistogram,
                        ComplexEnergyHistogram,
                        FormFactorsData,
                    ),
                )
                self.assertEqual(
                    tuple(buffer._file_name for buffer in simulation.raw_buffers),
                    (
                        "resonance_histogram",
                        "width_histogram",
                        "resonance_spacing_histogram",
                        "complex_energy_histogram",
                        "resonance_form_factors",
                    ),
                )
                self.assertEqual(
                    tuple(buffer.metadata["degree"] for buffer in coefficient_buffers),
                    tuple(range(1, max_degree + 1)),
                )
                self.assertTrue(
                    all(
                        isinstance(buffer, ResonanceCoefficientsHistogram)
                        for buffer in coefficient_buffers
                    )
                )
                self.assertEqual(
                    tuple(buffer.polynomial_degree for buffer in averaged_buffers),
                    tuple(range(1, max_degree + 1)),
                )
                self.assertEqual(
                    tuple(buffer.polynomial_degree for buffer in variate_buffers),
                    tuple(range(1, max_degree + 1)),
                )
                self.assertTrue(
                    all(
                        data.metadata.get("unfolding") == "raw"
                        for data in simulation.raw_buffers
                    )
                )
                self.assertTrue(
                    all(
                        data.metadata["unfolding"] == "weight"
                        for data in simulation.wgt_unfolded_buffers
                    )
                )
                for polynomial_degree, buffers in enumerate(averaged_buffers, start=1):
                    self.assertTrue(
                        all(
                            data.metadata
                            == {
                                "unfolding": "averaged",
                                "polynomial_degree": polynomial_degree,
                            }
                            for data in buffers
                        )
                    )
                for polynomial_degree, buffers in enumerate(variate_buffers, start=1):
                    self.assertTrue(
                        all(
                            data.metadata
                            == {
                                "unfolding": "variate",
                                "polynomial_degree": polynomial_degree,
                            }
                            for data in buffers
                        )
                    )

                if max_degree:
                    self.assertEqual(
                        tuple(data._file_name for data in averaged_buffers[0]),
                        (
                            "resonance_histogram_averaged_unfolded_degree_1",
                            "width_histogram_averaged_unfolded_degree_1",
                            "resonance_spacing_histogram_averaged_unfolded_degree_1",
                            "complex_energy_histogram_averaged_unfolded_degree_1",
                            "resonance_form_factors_averaged_unfolded_degree_1",
                        ),
                    )

                self.assertFalse(hasattr(simulation, "request"))
                self.assertFalse(hasattr(simulation, "result"))

    def test_poles_are_accumulated_and_finalized_in_raw_and_weight_buffers(
        self,
    ) -> None:
        simulation = ResonanceStatisticsSimulation(
            compound=build_compound(),
            realizs=1,
        )
        poles = np.array([-0.5 - 0.1j, 0.25 - 0.4j], dtype=np.complex128)
        resonance_centers = poles.real
        resonance_widths = -2 * poles.imag
        ensemble = simulation.compound.ensemble

        with patch.object(
            CompoundEnsemble,
            "resonances_stream",
            return_value=iter((poles,)),
        ):
            returned = simulation.execute()

        self.assertIsNone(returned)
        self.assertEqual(simulation.execution_state, ExecutionState.COMPLETE)

        raw = simulation.raw_buffers
        np.testing.assert_array_equal(
            raw.resonance_centers.counts,
            histogram_counts((resonance_centers,), bins=raw.resonance_centers.bins),
        )
        np.testing.assert_array_equal(
            raw.resonance_widths.counts,
            histogram_counts(
                (resonance_widths / ensemble.spectral_radius,),
                bins=raw.resonance_widths.bins,
            ),
        )
        np.testing.assert_array_equal(
            raw.nn_spacings.counts,
            histogram_counts((np.array([0.75]),), bins=raw.nn_spacings.bins),
        )
        np.testing.assert_array_equal(
            raw.complex_energies.counts,
            histogram2d_counts(
                resonance_centers / ensemble.spectral_radius,
                resonance_widths / ensemble.spectral_radius,
                histogram=raw.complex_energies,
            ),
        )
        self.assertAlmostEqual(float(np.sum(raw.complex_energies.histogram)), 1.0)

        raw_moment = np.mean(
            np.exp(-1j * np.outer(resonance_centers, raw.form_factors.times)),
            axis=0,
        )
        np.testing.assert_allclose(raw.form_factors.first_moment, raw_moment)
        np.testing.assert_allclose(
            raw.form_factors.form_factor,
            np.abs(raw_moment) ** 2,
        )
        np.testing.assert_allclose(raw.form_factors.connected_form_factor, 0.0)

        weight_cdf = simulation.compound.resonance_density.weight_cdf
        unfolded_centers = unfold_values(
            resonance_centers,
            cdf=weight_cdf,
            dimension=ensemble.dimension,
        )
        unfolded_widths = unfold_widths(
            resonance_widths,
            centers=resonance_centers,
            cdf=weight_cdf,
            dimension=ensemble.dimension,
        )
        weight = simulation.wgt_unfolded_buffers
        np.testing.assert_array_equal(
            weight.resonance_centers.counts,
            histogram_counts((unfolded_centers,), bins=weight.resonance_centers.bins),
        )
        np.testing.assert_array_equal(
            weight.resonance_widths.counts,
            histogram_counts((unfolded_widths,), bins=weight.resonance_widths.bins),
        )
        np.testing.assert_array_equal(
            weight.complex_energies.counts,
            histogram2d_counts(
                resonance_centers / ensemble.spectral_radius,
                unfolded_widths,
                histogram=weight.complex_energies,
            ),
        )
        self.assertAlmostEqual(float(np.sum(weight.complex_energies.histogram)), 1.0)

        for data in simulation:
            if isinstance(data, (ResonanceHistogram, WidthHistogram)) and np.sum(
                data.counts
            ):
                np.testing.assert_allclose(
                    data.histogram,
                    data.counts / (np.sum(data.counts) * np.diff(data.bins)),
                )

    def test_seeded_sampling_calibration_and_unfolding_match_control(self) -> None:
        simulation = ResonanceStatisticsSimulation(
            compound=build_compound(max_degree=2, seed=314159),
            realizs=2,
        )
        control = build_compound(max_degree=2, seed=314159)
        initial_rng_state = deepcopy(simulation.compound.rng_state)

        control_factory = TruncatedPolynomialCDFFactory(
            density=control.resonance_density,
            degrees=(1, 2),
            density_name="resonance",
        )
        average_cdfs = control_factory.average_interpolators()
        poles_samples = tuple(control.resonances_stream(realizs=2))
        resonance_center_samples = tuple(poles.real for poles in poles_samples)
        resonance_width_samples = tuple(-2 * poles.imag for poles in poles_samples)

        simulation.execute()

        self.assertEqual(simulation.compound.rng_state, control.rng_state)
        self.assertEqual(simulation.manifest.rng["initial_state"], initial_rng_state)
        self.assertEqual(
            simulation.manifest.rng["final_state"],
            control.rng_state,
        )
        calibration = cast(
            dict[str, object], simulation.manifest.execution["calibration"]
        )
        self.assertEqual(calibration["timing"], "cached_during_execution")

        coefficient_samples = tuple(
            control.resonance_density.compute_variate_coeffs(resonance_centers)
            for resonance_centers in resonance_center_samples
        )
        for degree, histogram in enumerate(simulation.coefficient_buffers, start=1):
            expected_samples = tuple(
                coefficients[degree : degree + 1] for coefficients in coefficient_samples
            )
            np.testing.assert_array_equal(
                histogram.counts,
                histogram_counts(expected_samples, bins=histogram.bins),
            )

        for cdf, buffers in zip(
            average_cdfs,
            simulation.ave_unfolded_buffers,
            strict=True,
        ):
            expected_centers = tuple(
                unfold_values(
                    resonance_centers,
                    cdf=cdf,
                    dimension=control.ensemble.dimension,
                )
                for resonance_centers in resonance_center_samples
            )
            expected_widths = tuple(
                unfold_widths(
                    resonance_widths,
                    centers=resonance_centers,
                    cdf=cdf,
                    dimension=control.ensemble.dimension,
                )
                for resonance_centers, resonance_widths in zip(
                    resonance_center_samples,
                    resonance_width_samples,
                    strict=True,
                )
            )
            np.testing.assert_array_equal(
                buffers.resonance_centers.counts,
                histogram_counts(expected_centers, bins=buffers.resonance_centers.bins),
            )
            np.testing.assert_array_equal(
                buffers.resonance_widths.counts,
                histogram_counts(expected_widths, bins=buffers.resonance_widths.bins),
            )

        variate_samples: list[list[np.ndarray]] = [[], []]
        for resonance_centers, resonance_widths, coefficients in zip(
            resonance_center_samples,
            resonance_width_samples,
            coefficient_samples,
            strict=True,
        ):
            variate_cdfs = control_factory.interpolators_from_coeffs(coefficients)
            for index, cdf in enumerate(variate_cdfs):
                variate_samples[index].append(
                    unfold_widths(
                        resonance_widths,
                        centers=resonance_centers,
                        cdf=cdf,
                        dimension=control.ensemble.dimension,
                    )
                )

        for buffers, unfolded_width_samples in zip(
            simulation.var_unfolded_buffers,
            variate_samples,
            strict=True,
        ):
            np.testing.assert_array_equal(
                buffers.resonance_widths.counts,
                histogram_counts(
                    tuple(unfolded_width_samples),
                    bins=buffers.resonance_widths.bins,
                ),
            )

    def test_failed_execution_is_terminal(self) -> None:
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
        with self.assertRaisesRegex(RuntimeError, "only after execution"):
            simulation.save()
        with self.assertRaisesRegex(RuntimeError, "only after execution"):
            simulation.plot(Path("unused"))

    def test_save_load_plot_dispatch_and_missing_data_validation(self) -> None:
        simulation = ResonanceStatisticsSimulation(
            compound=build_compound(max_degree=1, seed=77),
            realizs=1,
        )
        simulation.execute()
        completed_rng_state = deepcopy(simulation.compound.rng_state)

        with tempfile.TemporaryDirectory() as temporary_directory:
            destination_directory = simulation.save(temporary_directory)
            restored_simulation = load_resonance_statistics_simulation(
                directory=destination_directory
            )

            self.assertEqual(
                restored_simulation.execution_state,
                ExecutionState.COMPLETE,
            )
            self.assertEqual(
                tuple(type(data) for data in restored_simulation),
                tuple(type(data) for data in simulation),
            )
            for restored_data, original_data in zip(
                restored_simulation,
                simulation,
                strict=True,
            ):
                self.assertEqual(restored_data.metadata, original_data.metadata)
                if isinstance(
                    restored_data,
                    (
                        ResonanceCoefficientsHistogram,
                        ResonanceHistogram,
                        WidthHistogram,
                        ResonanceSpacingHistogram,
                        ComplexEnergyHistogram,
                    ),
                ):
                    original_histogram = cast(
                        ResonanceHistogramData,
                        original_data,
                    )
                    np.testing.assert_array_equal(
                        restored_data.counts,
                        original_histogram.counts,
                    )
                    np.testing.assert_allclose(
                        restored_data.histogram,
                        original_histogram.histogram,
                    )

            np.testing.assert_array_equal(
                restored_simulation.compound.resonance_density.average_coeffs,
                simulation.compound.resonance_density.average_coeffs,
            )

            plot_patchers = [
                patch.object(plot_cls, "plot", autospec=True)
                for plot_cls in RESONANCE_PLOT_CLASSES
            ]
            plot_mocks = [plot_patcher.start() for plot_patcher in plot_patchers]
            try:
                plot_resonance_statistics_simulation(directory=destination_directory)
            finally:
                for plot_patcher in reversed(plot_patchers):
                    plot_patcher.stop()

            expected_call_counts = (1, 1, 3, 1, 3, 1, 3, 3, 1, 1, 3)
            for plot_mock, expected_call_count in zip(
                plot_mocks,
                expected_call_counts,
                strict=True,
            ):
                self.assertEqual(plot_mock.call_count, expected_call_count)
            self.assertIsNotNone(plot_mocks[1].call_args.args[0].context)

            manifest_path = destination_directory / "manifest.json"
            original_manifest_text = manifest_path.read_text(encoding="utf-8")
            manifest = json.loads(original_manifest_text)
            manifest["execution"]["calibration"]["average_coefficients"] = []
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "invalid shape"):
                load_resonance_statistics_simulation(directory=destination_directory)
            manifest_path.write_text(original_manifest_text, encoding="utf-8")

            unexpected = ResonanceHistogram(
                _file_name="unexpected_resonance_histogram",
                support=(-1.0, 1.0),
            )
            unexpected.save(directory=destination_directory)
            with self.assertRaisesRegex(ValueError, "not part of the simulation"):
                load_resonance_statistics_simulation(directory=destination_directory)
            (destination_directory / unexpected.to_path).unlink()

            missing_data_path = (
                destination_directory / next(iter(restored_simulation)).to_path
            )
            missing_data_path.unlink()
            with self.assertRaisesRegex(ValueError, "Saved data .* is missing"):
                load_resonance_statistics_simulation(directory=destination_directory)
            with self.assertRaisesRegex(ValueError, "Saved data .* is missing"):
                restored_simulation.plot(destination_directory)

        self.assertEqual(simulation.compound.rng_state, completed_rng_state)

    def test_plot_configuration_uses_a_detached_compound(self) -> None:
        simulation = ResonanceStatisticsSimulation(
            compound=build_compound(seed=902),
            realizs=1,
        )
        simulation.execute()
        completed_rng_state = deepcopy(simulation.compound.rng_state)

        plot = ComplexEnergyHistogramPlot(
            data=simulation.raw_buffers.complex_energies,
            context=simulation.manifest,
        )
        plot.set_derived_attributes()

        self.assertIsNot(plot.compound, simulation.compound)
        self.assertEqual(simulation.compound.rng_state, completed_rng_state)

    def test_run_helper_executes_saves_reloads_and_dispatches_plots(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            ExitStack() as plot_stack,
        ):
            plot_mocks = [
                plot_stack.enter_context(patch.object(plot_cls, "plot", autospec=True))
                for plot_cls in RESONANCE_PLOT_CLASSES
            ]
            simulation = run_resonance_statistics_simulation(
                compound=build_compound(seed=311),
                realizs=1,
                directory=temporary_directory,
            )

            self.assertEqual(simulation.execution_state, ExecutionState.COMPLETE)
            self.assertEqual(plot_mocks[0].call_count, 0)
            for plot_mock in plot_mocks[1:]:
                self.assertEqual(plot_mock.call_count, 1)

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
