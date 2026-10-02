import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
from scipy.special import jn_zeros

from rmtpy.compounds import CompoundEnsemble
from rmtpy.ensembles import GOE
from rmtpy.simulations.base_simulation import ExecutionState
from rmtpy.simulations.time_delay_statistics import (
    TimeDelayStatisticsSimulation,
    load_time_delay_statistics_simulation,
    plot_time_delay_statistics_simulation,
    run_time_delay_statistics_simulation,
)
from rmtpy.simulations.time_delay_statistics.time_delay_histogram import (
    TimeDelayHistogram,
    TimeDelayHistogramPlot,
    UnfoldedTimeDelayHistogramPlot,
)
from rmtpy.simulations.time_delay_statistics.time_delay_statistics_simulation import (
    unfold_time_delays,
)
from rmtpy.simulations.unfolding import TruncatedPolynomialCDFFactory


def build_compound(*, max_degree: int = 0, seed: int = 123) -> CompoundEnsemble:
    return CompoundEnsemble(
        ensemble=GOE(
            num_majoranas=4,
            max_spectral_polynomial_degree=max_degree,
            seed=seed,
        ),
        couplings=np.array([0.75, 1.25]),
    )


def time_delay_histograms(
    simulation: TimeDelayStatisticsSimulation,
) -> tuple[TimeDelayHistogram, ...]:
    return cast(tuple[TimeDelayHistogram, ...], tuple(simulation))


def as_energy_argument(
    value: object,
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    return cast(np.ndarray[tuple[int], np.dtype[np.floating]], value)


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


class TimeDelayStatisticsTests(unittest.TestCase):
    def test_required_energies_are_normalized_copied_and_validated(self) -> None:
        source = np.array([-0.25, 0.125])
        simulation = TimeDelayStatisticsSimulation(
            compound=build_compound(),
            energies=source,
            realizs=1,
        )
        source[0] = -0.5

        np.testing.assert_array_equal(simulation.energies, np.array([-0.25, 0.125]))
        self.assertFalse(np.shares_memory(source, simulation.energies))
        self.assertFalse(simulation.energies.flags.writeable)
        with self.assertRaises(ValueError):
            simulation.energies[0] = 0.0

        scalar = TimeDelayStatisticsSimulation(
            compound=build_compound(),
            energies=as_energy_argument(0.25),
            realizs=1,
        )
        np.testing.assert_array_equal(scalar.energies, np.array([0.25]))

        nearby = TimeDelayStatisticsSimulation(
            compound=build_compound(),
            energies=as_energy_argument((0.123456, 0.123457)),
            realizs=1,
        )
        np.testing.assert_array_equal(nearby.energies, [0.123456, 0.123457])

        invalid_energies = (
            (),
            ((0.0, 0.1),),
            (np.nan,),
            (np.inf,),
            (-0.0, 0.0),
            (0.1, 0.1),
        )
        for energies in invalid_energies:
            with self.subTest(energies=energies), self.assertRaises(ValueError):
                TimeDelayStatisticsSimulation(
                    compound=build_compound(),
                    energies=as_energy_argument(energies),
                    realizs=1,
                )

        with self.assertRaises(TypeError):
            TimeDelayStatisticsSimulation(  # pyright: ignore[reportCallIssue]
                compound=build_compound(),
                realizs=1,
            )

    def test_buffer_schema_filenames_metadata_and_iteration_order(self) -> None:
        energies = (-0.25, 0.125)
        expected_counts = {0: 4, 2: 12}
        for max_degree, expected_count in expected_counts.items():
            with self.subTest(max_degree=max_degree):
                simulation = TimeDelayStatisticsSimulation(
                    compound=build_compound(max_degree=max_degree),
                    energies=as_energy_argument(energies),
                    realizs=1,
                )

                averaged_buffers = tuple(simulation.ave_unfolded_buffers)
                variate_buffers = tuple(simulation.var_unfolded_buffers)
                data = tuple(simulation)

                self.assertEqual(len(data), expected_count)
                self.assertTrue(
                    all(isinstance(histogram, TimeDelayHistogram) for histogram in data)
                )
                self.assertEqual(
                    tuple(histogram._file_name for histogram in simulation.raw_buffers),
                    (
                        "time_delay_energy_0_histogram",
                        "time_delay_energy_1_histogram",
                    ),
                )
                self.assertEqual(
                    tuple(
                        histogram._file_name
                        for histogram in simulation.wgt_unfolded_buffers
                    ),
                    (
                        "time_delay_energy_0_histogram_weight_unfolded",
                        "time_delay_energy_1_histogram_weight_unfolded",
                    ),
                )
                self.assertEqual(
                    tuple(buffer.polynomial_degree for buffer in averaged_buffers),
                    tuple(range(1, max_degree + 1)),
                )
                self.assertEqual(
                    tuple(buffer.polynomial_degree for buffer in variate_buffers),
                    tuple(range(1, max_degree + 1)),
                )

                first_bessel_zero = cast(float, jn_zeros(1, 1)[0])
                raw_scale = (
                    first_bessel_zero / simulation.compound.ensemble.spectral_radius
                )
                for energy_index, histogram in enumerate(
                    simulation.raw_buffers.time_delays
                ):
                    self.assertEqual(
                        histogram.metadata,
                        {
                            "energy_index": energy_index,
                            "energy": energies[energy_index],
                            "scale": raw_scale,
                            "unfolding": "raw",
                        },
                    )
                    self.assertEqual(
                        histogram.log_base,
                        simulation.compound.ensemble.dimension,
                    )
                    self.assertEqual(histogram.num_bins, 100)

                for energy_index, histogram in enumerate(
                    simulation.wgt_unfolded_buffers.time_delays
                ):
                    self.assertEqual(
                        histogram.metadata,
                        {
                            "energy_index": energy_index,
                            "energy": energies[energy_index],
                            "scale": 2 * np.pi,
                            "unfolding": "weight",
                        },
                    )

                for polynomial_degree, buffers in enumerate(
                    averaged_buffers,
                    start=1,
                ):
                    for energy_index, histogram in enumerate(buffers):
                        self.assertEqual(
                            histogram.metadata,
                            {
                                "energy_index": energy_index,
                                "energy": energies[energy_index],
                                "scale": 2 * np.pi,
                                "unfolding": "averaged",
                                "polynomial_degree": polynomial_degree,
                            },
                        )

                if max_degree:
                    self.assertEqual(
                        tuple(histogram._file_name for histogram in averaged_buffers[0]),
                        (
                            "time_delay_energy_0_histogram_averaged_unfolded_degree_1",
                            "time_delay_energy_1_histogram_averaged_unfolded_degree_1",
                        ),
                    )
                    self.assertEqual(
                        tuple(histogram._file_name for histogram in variate_buffers[0]),
                        (
                            "time_delay_energy_0_histogram_variate_unfolded_degree_1",
                            "time_delay_energy_1_histogram_variate_unfolded_degree_1",
                        ),
                    )

                self.assertFalse(hasattr(simulation, "request"))
                self.assertFalse(hasattr(simulation, "result"))

    def test_reciprocal_width_unfolding_and_sample_validation(self) -> None:
        time_delays = np.array([2.0, 4.0, np.nan, np.inf, 0.0, -1.0])
        unfolded = unfold_time_delays(
            time_delays,
            energy=0.25,
            cdf=lambda values: values,
            dimension=2,
        )
        np.testing.assert_allclose(unfolded, np.array([1.0, 2.0]))

        empty = unfold_time_delays(
            np.array([0.0, -1.0, np.nan]),
            energy=0.0,
            cdf=lambda values: values,
            dimension=2,
        )
        self.assertEqual(empty.size, 0)

        simulation = TimeDelayStatisticsSimulation(
            compound=build_compound(),
            energies=as_energy_argument((-0.1, 0.1)),
            realizs=1,
        )
        malformed = np.ones((2, 1))
        with (
            patch.object(
                CompoundEnsemble,
                "time_delays_stream",
                return_value=iter(((malformed, np.array([-0.5, 0.5])),)),
            ),
            self.assertRaisesRegex(ValueError, "must have shape"),
        ):
            simulation.execute()

        self.assertEqual(simulation.execution_state, ExecutionState.FAILED)
        with self.assertRaisesRegex(RuntimeError, "only once"):
            simulation.execute()
        with self.assertRaisesRegex(RuntimeError, "only after execution"):
            simulation.save()
        with self.assertRaisesRegex(RuntimeError, "only after execution"):
            simulation.plot(Path("unused"))

    def test_raw_and_weight_delays_are_accumulated_and_finalized(self) -> None:
        simulation = TimeDelayStatisticsSimulation(
            compound=build_compound(max_degree=1),
            energies=as_energy_argument((-0.1, 0.1)),
            realizs=1,
        )
        time_delays = np.array([[1.0, 2.0], [3.0, 4.0]])
        closed_eigenvalues = np.array([-0.75, 0.5])
        spectral_density = simulation.compound.ensemble.spectral_density
        compute_coefficients = spectral_density.compute_variate_coeffs

        with (
            patch.object(
                CompoundEnsemble,
                "time_delays_stream",
                return_value=iter(((time_delays, closed_eigenvalues),)),
            ),
            patch.object(
                type(spectral_density),
                "compute_variate_coeffs",
                autospec=True,
                side_effect=lambda _density, values: compute_coefficients(values),
            ) as compute,
        ):
            returned = simulation.execute()

        self.assertIsNone(returned)
        self.assertEqual(simulation.execution_state, ExecutionState.COMPLETE)
        np.testing.assert_array_equal(compute.call_args.args[1], closed_eigenvalues)

        for energy_index, histogram in enumerate(simulation.raw_buffers.time_delays):
            np.testing.assert_array_equal(
                histogram.counts,
                histogram_counts(
                    (time_delays[energy_index],),
                    bins=histogram.bins,
                ),
            )

        weight_cdf = spectral_density.weight_cdf
        for energy_index, histogram in enumerate(
            simulation.wgt_unfolded_buffers.time_delays
        ):
            unfolded_delays = unfold_time_delays(
                time_delays[energy_index],
                energy=float(simulation.energies[energy_index]),
                cdf=weight_cdf,
                dimension=simulation.compound.ensemble.dimension,
            )
            np.testing.assert_array_equal(
                histogram.counts,
                histogram_counts((unfolded_delays,), bins=histogram.bins),
            )

        for histogram in time_delay_histograms(simulation):
            if np.sum(histogram.counts):
                np.testing.assert_allclose(
                    histogram.histogram,
                    histogram.counts
                    / (np.sum(histogram.counts) * np.diff(histogram.bins)),
                )

    def test_seeded_sampling_calibration_and_unfolding_match_control(self) -> None:
        simulation = TimeDelayStatisticsSimulation(
            compound=build_compound(max_degree=2, seed=314159),
            energies=as_energy_argument((-0.1, 0.1)),
            realizs=2,
        )
        control = build_compound(max_degree=2, seed=314159)
        initial_rng_state = deepcopy(simulation.compound.rng_state)

        cdf_factory = TruncatedPolynomialCDFFactory(
            density=control.ensemble.spectral_density,
            degrees=(1, 2),
            density_name="spectral",
        )
        average_cdfs = cdf_factory.average_interpolators()
        samples: list[np.ndarray] = []
        variate_cdfs_by_sample = []
        for time_delays, closed_eigenvalues in control.time_delays_stream(
            energies=simulation.energies,
            realizs=2,
        ):
            samples.append(time_delays.copy())
            coefficients = control.ensemble.spectral_density.compute_variate_coeffs(
                closed_eigenvalues
            )
            variate_cdfs_by_sample.append(
                cdf_factory.interpolators_from_coeffs(coefficients)
            )

        simulation.execute()

        self.assertEqual(simulation.compound.rng_state, control.rng_state)
        self.assertEqual(simulation.manifest.rng["initial_state"], initial_rng_state)
        self.assertEqual(simulation.manifest.rng["final_state"], control.rng_state)
        calibration = cast(
            dict[str, object], simulation.manifest.execution["calibration"]
        )
        self.assertEqual(calibration["timing"], "cached_during_execution")

        for energy_index, energy in enumerate(simulation.energies):
            raw_samples = tuple(sample[energy_index] for sample in samples)
            raw_histogram = simulation.raw_buffers.time_delays[energy_index]
            np.testing.assert_array_equal(
                raw_histogram.counts,
                histogram_counts(raw_samples, bins=raw_histogram.bins),
            )

            for cdf, buffers in zip(
                average_cdfs,
                simulation.ave_unfolded_buffers,
                strict=True,
            ):
                average_samples = tuple(
                    unfold_time_delays(
                        sample[energy_index],
                        energy=float(energy),
                        cdf=cdf,
                        dimension=control.ensemble.dimension,
                    )
                    for sample in samples
                )
                average_histogram = buffers.time_delays[energy_index]
                np.testing.assert_array_equal(
                    average_histogram.counts,
                    histogram_counts(
                        average_samples,
                        bins=average_histogram.bins,
                    ),
                )

            for degree_index, buffers in enumerate(simulation.var_unfolded_buffers):
                variate_samples = tuple(
                    unfold_time_delays(
                        sample[energy_index],
                        energy=float(energy),
                        cdf=cdfs[degree_index],
                        dimension=control.ensemble.dimension,
                    )
                    for sample, cdfs in zip(
                        samples,
                        variate_cdfs_by_sample,
                        strict=True,
                    )
                )
                variate_histogram = buffers.time_delays[energy_index]
                np.testing.assert_array_equal(
                    variate_histogram.counts,
                    histogram_counts(
                        variate_samples,
                        bins=variate_histogram.bins,
                    ),
                )

    def test_save_load_plot_dispatch_and_archive_validation(self) -> None:
        simulation = TimeDelayStatisticsSimulation(
            compound=build_compound(max_degree=1, seed=77),
            energies=as_energy_argument((0.123456, 0.123457)),
            realizs=1,
        )
        simulation.execute()
        completed_rng_state = deepcopy(simulation.compound.rng_state)

        with tempfile.TemporaryDirectory() as temporary_directory:
            destination_directory = simulation.save(temporary_directory)
            restored_simulation = load_time_delay_statistics_simulation(
                directory=destination_directory
            )

            self.assertEqual(
                restored_simulation.execution_state,
                ExecutionState.COMPLETE,
            )
            np.testing.assert_array_equal(
                restored_simulation.energies,
                simulation.energies,
            )
            self.assertFalse(restored_simulation.energies.flags.writeable)
            self.assertEqual(
                tuple(type(histogram) for histogram in restored_simulation),
                tuple(type(histogram) for histogram in simulation),
            )
            for restored_histogram, original_histogram in zip(
                time_delay_histograms(restored_simulation),
                time_delay_histograms(simulation),
                strict=True,
            ):
                self.assertEqual(
                    restored_histogram.metadata,
                    original_histogram.metadata,
                )
                np.testing.assert_array_equal(
                    restored_histogram.counts,
                    original_histogram.counts,
                )
                np.testing.assert_allclose(
                    restored_histogram.histogram,
                    original_histogram.histogram,
                )

            np.testing.assert_array_equal(
                restored_simulation.compound.ensemble.spectral_density.average_coeffs,
                simulation.compound.ensemble.spectral_density.average_coeffs,
            )

            with (
                patch.object(
                    TimeDelayHistogramPlot,
                    "plot",
                    autospec=True,
                ) as raw_plot,
                patch.object(
                    UnfoldedTimeDelayHistogramPlot,
                    "plot",
                    autospec=True,
                ) as unfolded_plot,
            ):
                plot_time_delay_statistics_simulation(directory=destination_directory)

            self.assertEqual(raw_plot.call_count, 2)
            self.assertEqual(unfolded_plot.call_count, 6)
            self.assertIsNotNone(raw_plot.call_args.args[0].context)

            manifest_path = destination_directory / "manifest.json"
            manifest_text = manifest_path.read_text(encoding="utf-8")
            manifest = json.loads(manifest_text)
            manifest["execution"]["calibration"]["average_coefficients"] = []
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "invalid shape"):
                load_time_delay_statistics_simulation(directory=destination_directory)
            manifest_path.write_text(manifest_text, encoding="utf-8")

            unexpected = TimeDelayHistogram(
                _file_name="unexpected_time_delay_histogram",
                support=(1.0, 2.0),
            )
            unexpected.save(directory=destination_directory)
            with self.assertRaisesRegex(ValueError, "not part of the simulation"):
                load_time_delay_statistics_simulation(directory=destination_directory)
            (destination_directory / unexpected.to_path).unlink()

            missing_data_path = (
                destination_directory / next(iter(restored_simulation)).to_path
            )
            missing_data_path.unlink()
            with self.assertRaisesRegex(ValueError, "Saved data .* is missing"):
                load_time_delay_statistics_simulation(directory=destination_directory)
            with self.assertRaisesRegex(ValueError, "Saved data .* is missing"):
                restored_simulation.plot(destination_directory)

        self.assertEqual(simulation.compound.rng_state, completed_rng_state)

    def test_plot_configuration_uses_a_detached_compound(self) -> None:
        simulation = TimeDelayStatisticsSimulation(
            compound=build_compound(max_degree=1, seed=902),
            energies=as_energy_argument((0.0,)),
            realizs=1,
        )
        simulation.execute()
        completed_rng_state = deepcopy(simulation.compound.rng_state)

        raw_plot = TimeDelayHistogramPlot(
            data=simulation.raw_buffers.time_delays[0],
            context=simulation.manifest,
        )
        unfolded_plot = UnfoldedTimeDelayHistogramPlot(
            data=simulation.wgt_unfolded_buffers.time_delays[0],
            context=simulation.manifest,
        )
        raw_plot.set_derived_attributes()
        unfolded_plot.set_derived_attributes()

        self.assertIsNot(raw_plot.compound, simulation.compound)
        self.assertIsNot(unfolded_plot.compound, simulation.compound)
        self.assertEqual(simulation.compound.rng_state, completed_rng_state)

    def test_run_helper_executes_saves_reloads_and_dispatches_plots(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            patch.object(
                TimeDelayHistogramPlot,
                "plot",
                autospec=True,
            ) as raw_plot,
            patch.object(
                UnfoldedTimeDelayHistogramPlot,
                "plot",
                autospec=True,
            ) as unfolded_plot,
        ):
            simulation = run_time_delay_statistics_simulation(
                compound=build_compound(seed=311),
                energies=as_energy_argument((0.0,)),
                realizs=1,
                directory=temporary_directory,
            )

            self.assertEqual(simulation.execution_state, ExecutionState.COMPLETE)
            raw_plot.assert_called_once()
            unfolded_plot.assert_called_once()
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
