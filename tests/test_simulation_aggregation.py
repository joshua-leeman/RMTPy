import json
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path

import numpy as np

from rmtpy.compounds import CompoundEnsemble
from rmtpy.conversion import unwrap_json_value
from rmtpy.ensembles import GOE
from rmtpy.simulations import (
    PartialWidthsStatisticsSimulation,
    ResonanceStatisticsSimulation,
    Simulation,
    SpectralStatisticsSimulation,
    TimeDelayStatisticsSimulation,
    TransmissionCoefficientsSimulation,
)
from rmtpy.simulations.base_simulation import ExecutionState
from rmtpy.simulations.spectral_statistics.spectral_coefficients_histogram import (
    SpectralCoefficientsHistogram,
)


def rewrite_data_archive(
    path: Path,
    *,
    omitted_fields: tuple[str, ...] = (),
    omitted_metadata: tuple[str, ...] = (),
) -> None:
    with np.load(path, allow_pickle=False) as archive:
        payload = {
            name: archive[name] for name in archive.files if name not in omitted_fields
        }

    if omitted_metadata:
        metadata = json.loads(str(payload["metadata"].item()))
        for name in omitted_metadata:
            metadata.pop(name, None)
        payload["metadata"] = np.asarray(json.dumps(metadata))

    np.savez(path, **payload)


def build_compound(*, seed: int) -> CompoundEnsemble:
    return CompoundEnsemble(
        ensemble=GOE(
            num_majoranas=4,
            max_spectral_polynomial_degree=0,
            seed=seed,
        ),
        couplings=np.array([1.0, 1.0]),
    )


def build_spectral(*, seed: int, realizs: int) -> Simulation:
    return SpectralStatisticsSimulation(
        ensemble=GOE(
            num_majoranas=4,
            max_spectral_polynomial_degree=0,
            seed=seed,
        ),
        realizs=realizs,
    )


def build_resonance(*, seed: int, realizs: int) -> Simulation:
    return ResonanceStatisticsSimulation(
        compound=build_compound(seed=seed),
        realizs=realizs,
    )


def build_partial_widths(*, seed: int, realizs: int) -> Simulation:
    return PartialWidthsStatisticsSimulation(
        compound=build_compound(seed=seed),
        width_indices=((0, 0), (0,)),
        realizs=realizs,
    )


def build_time_delays(*, seed: int, realizs: int) -> Simulation:
    return TimeDelayStatisticsSimulation(
        compound=build_compound(seed=seed),
        energies=np.array([-0.25, 0.25]),
        realizs=realizs,
    )


def build_transmission(*, seed: int, realizs: int) -> Simulation:
    return TransmissionCoefficientsSimulation(
        compound=build_compound(seed=seed),
        channel_indices=(0,),
        realizs=realizs,
    )


SIMULATION_BUILDERS: tuple[Callable[..., Simulation], ...] = (
    build_spectral,
    build_resonance,
    build_partial_widths,
    build_time_delays,
    build_transmission,
)


class SimulationAggregationTests(unittest.TestCase):
    def test_all_simulation_families_aggregate_and_round_trip(self) -> None:
        additive_fields = (
            "counts",
            "first_moment",
            "second_moment",
            "scattering_diagonal_sum",
        )

        for builder in SIMULATION_BUILDERS:
            with (
                self.subTest(builder=builder.__name__),
                tempfile.TemporaryDirectory() as root,
            ):
                superfolder = Path(root)
                sources: list[Simulation] = []
                source_directories: list[Path] = []
                for job_index, (seed, realizs) in enumerate(((101, 1), (202, 2))):
                    source = builder(seed=seed, realizs=realizs)
                    source.execute()
                    source_directory = source.save(
                        superfolder / f"job_{job_index}_outputs"
                    )
                    sources.append(source)
                    source_directories.append(source_directory)

                aggregate = type(sources[0]).aggregate(superfolder)

                self.assertIsInstance(aggregate, type(sources[0]))
                self.assertEqual(aggregate.execution_state, ExecutionState.COMPLETE)
                self.assertEqual(aggregate.realizs, 3)
                self.assertIsNone(aggregate._rmg.seed)
                self.assertEqual(aggregate.manifest.rng["policy"], "aggregate")
                self.assertEqual(
                    aggregate.manifest.execution["aggregation"]["job_indices"],
                    [0, 1],
                )
                self.assertEqual(
                    aggregate.manifest.execution["aggregation"]["source_realizs"],
                    [1, 2],
                )
                self.assertEqual(
                    aggregate.manifest.rng["seed"],
                    {"type": "aggregate", "source_seeds": [101, 202]},
                )
                source_completion_times = aggregate.manifest.execution["aggregation"][
                    "source_completion_times"
                ]
                self.assertTrue(
                    all(
                        isinstance(completion_time, str)
                        for completion_time in source_completion_times
                    )
                )

                aggregate_data = {data._file_name: data for data in aggregate}
                source_data = [
                    {data._file_name: data for data in source} for source in sources
                ]
                for file_name, combined in aggregate_data.items():
                    self.assertEqual(combined.realizs, 3)
                    for field_name in additive_fields:
                        if not hasattr(combined, field_name):
                            continue
                        expected = sum(
                            (
                                getattr(items[file_name], field_name)
                                for items in source_data
                            ),
                            start=np.zeros_like(getattr(combined, field_name)),
                        )
                        np.testing.assert_allclose(
                            getattr(combined, field_name),
                            expected,
                        )
                    if hasattr(combined, "sample_sum"):
                        self.assertAlmostEqual(
                            combined.sample_sum,
                            sum(items[file_name].sample_sum for items in source_data),
                        )
                    if hasattr(combined, "single_realization_form_factor"):
                        np.testing.assert_array_equal(
                            combined.single_realization_form_factor,
                            source_data[0][file_name].single_realization_form_factor,
                        )

                saved_aggregate = aggregate.save(superfolder / "aggregated_outputs")
                restored = Simulation.load(saved_aggregate)
                self.assertEqual(restored.execution_state, ExecutionState.COMPLETE)
                self.assertEqual(restored.realizs, 3)
                self.assertEqual(
                    restored.manifest.execution["aggregation"],
                    aggregate.manifest.execution["aggregation"],
                )
                restored_data = {data._file_name: data for data in restored}
                for file_name, combined in aggregate_data.items():
                    if hasattr(combined, "single_realization_form_factor"):
                        np.testing.assert_array_equal(
                            restored_data[file_name].single_realization_form_factor,
                            combined.single_realization_form_factor,
                        )
                for source_directory in source_directories:
                    self.assertTrue(source_directory.is_dir())

    def test_density_calibrations_are_pooled_by_realization_count(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            superfolder = Path(root)
            for job_index, (seed, realizs, coefficient) in enumerate(
                ((11, 1, 1.25), (22, 3, 2.75))
            ):
                simulation = build_spectral(seed=seed, realizs=realizs)
                object.__setattr__(
                    simulation.ensemble.spectral_density,
                    "average_coeffs",
                    np.array([coefficient], dtype=np.float64),
                )
                simulation.execute()
                simulation.save(superfolder / f"job_{job_index}_outputs")

            aggregate = SpectralStatisticsSimulation.aggregate(superfolder)
            calibration = aggregate.manifest.execution["calibration"]
            self.assertEqual(calibration["density"], "spectral")
            self.assertEqual(calibration["timing"], "aggregated")
            expected = np.array([(1.25 + 3 * 2.75) / 4], dtype=np.float64)
            np.testing.assert_allclose(
                unwrap_json_value(calibration["average_coefficients"]),
                expected,
            )
            np.testing.assert_allclose(
                aggregate.ensemble.spectral_density.average_coeffs,
                expected,
            )

    def test_legacy_width_archives_recover_their_additive_sample_sum(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            superfolder = Path(root)
            expected_sums: dict[str, float] = {}
            for job_index, seed in enumerate((31, 32)):
                simulation = build_partial_widths(seed=seed, realizs=2)
                simulation.execute()
                for data in simulation:
                    expected_sums[data._file_name] = expected_sums.get(
                        data._file_name, 0.0
                    ) + float(data.sample_sum)

                source_directory = simulation.save(
                    superfolder / f"job_{job_index}_outputs"
                )
                for archive_path in source_directory.rglob("*_data.npz"):
                    rewrite_data_archive(
                        archive_path,
                        omitted_fields=("sample_sum",),
                    )

            aggregate = PartialWidthsStatisticsSimulation.aggregate(superfolder)
            for data in aggregate:
                self.assertAlmostEqual(
                    data.sample_sum,
                    expected_sums[data._file_name],
                )
                self.assertAlmostEqual(
                    data.metadata["average_width"],
                    expected_sums[data._file_name] / aggregate.realizs,
                )

    def test_concrete_discovery_is_strict_and_configuration_aware(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            superfolder = Path(root)
            for job_index, num_majoranas in enumerate((4, 6)):
                simulation = SpectralStatisticsSimulation(
                    ensemble=GOE(
                        num_majoranas=num_majoranas,
                        max_spectral_polynomial_degree=0,
                        seed=job_index,
                    ),
                    realizs=1,
                )
                simulation.execute()
                simulation.save(superfolder / f"job_{job_index}_outputs")

            with self.assertRaisesRegex(ValueError, "scientific configuration"):
                SpectralStatisticsSimulation.aggregate(superfolder)

        with tempfile.TemporaryDirectory() as root:
            superfolder = Path(root)
            for job_index in range(2):
                job_directory = superfolder / f"job_{job_index}_outputs"
                for seed in (job_index, job_index + 10):
                    simulation = build_spectral(seed=seed, realizs=1)
                    simulation.execute()
                    simulation.save(job_directory)

            with self.assertRaisesRegex(ValueError, "exactly one"):
                SpectralStatisticsSimulation.aggregate(superfolder)

    def test_base_discovery_requires_an_unambiguous_simulation_class(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            superfolder = Path(root)
            job_directory = superfolder / "job_0_outputs"
            for simulation in (
                build_spectral(seed=1, realizs=1),
                build_transmission(seed=1, realizs=1),
            ):
                simulation.execute()
                simulation.save(job_directory)

            with self.assertRaisesRegex(ValueError, "unambiguous simulation"):
                Simulation.aggregate(superfolder)

    def test_concrete_discovery_rejects_wrong_classes_and_incomplete_runs(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            superfolder = Path(root)
            simulation = build_transmission(seed=1, realizs=1)
            simulation.execute()
            simulation.save(superfolder / "job_0_outputs")
            with self.assertRaisesRegex(ValueError, "exactly one"):
                SpectralStatisticsSimulation.aggregate(superfolder)

        with tempfile.TemporaryDirectory() as root:
            superfolder = Path(root)
            simulation = build_spectral(seed=1, realizs=1)
            simulation.execute()
            source_directory = simulation.save(superfolder / "job_0_outputs")
            manifest_path = source_directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["execution"]["execution_state"] = "failed"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not complete"):
                SpectralStatisticsSimulation.aggregate(superfolder)

    def test_coefficient_grids_and_tail_counts_are_exactly_additive(self) -> None:
        first = SpectralCoefficientsHistogram.create(degree=2, dimension=16)
        second = SpectralCoefficientsHistogram.create(degree=2, dimension=16)
        aggregate = SpectralCoefficientsHistogram.create(degree=2, dimension=16)

        np.testing.assert_array_equal(first.bins, second.bins)
        different_configuration = SpectralCoefficientsHistogram.create(
            degree=3,
            dimension=16,
        )
        self.assertFalse(np.array_equal(first.bins, different_configuration.bins))
        high_dimension = SpectralCoefficientsHistogram.create(
            degree=2,
            dimension=10_201,
        )
        self.assertEqual(high_dimension.num_bins, 101)
        first.add_histogram_contribution(np.array([-10.0, 0.0]))
        second.add_histogram_contribution(np.array([0.1, 10.0]))
        aggregate.add_contribution(first)
        aggregate.add_contribution(second)
        aggregate.compute_statistics()

        np.testing.assert_array_equal(aggregate.counts, first.counts + second.counts)
        self.assertEqual(aggregate.realizs, 2)
        self.assertEqual(aggregate.underflow, 1)
        self.assertEqual(aggregate.overflow, 1)

        legacy = SpectralCoefficientsHistogram.create(degree=2, dimension=16)
        legacy.metadata.pop("grid_policy")
        with self.assertRaisesRegex(ValueError, "cannot be aggregated exactly"):
            aggregate.add_contribution(legacy)

    def test_legacy_coefficient_archive_loads_but_cannot_be_aggregated(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            superfolder = Path(root)
            simulation = SpectralStatisticsSimulation(
                ensemble=GOE(
                    num_majoranas=4,
                    max_spectral_polynomial_degree=1,
                    seed=41,
                ),
                realizs=1,
            )
            object.__setattr__(
                simulation.ensemble.spectral_density,
                "average_coeffs",
                np.array([1.0, 0.0], dtype=np.float64),
            )
            simulation.execute()
            source_directory = simulation.save(superfolder / "job_0_outputs")
            coefficient_archive = next(
                source_directory.rglob("spectral_coeff_1_histogram_data.npz")
            )
            rewrite_data_archive(
                coefficient_archive,
                omitted_fields=("underflow", "overflow"),
                omitted_metadata=("grid_policy", "dimension"),
            )

            restored = Simulation.load(source_directory)
            coefficient_data = next(iter(restored.coefficient_buffers))
            self.assertEqual(coefficient_data.underflow, 0)
            self.assertEqual(coefficient_data.overflow, 0)
            with self.assertRaisesRegex(ValueError, "cannot be aggregated exactly"):
                SpectralStatisticsSimulation.aggregate(superfolder)

    def test_missing_job_outputs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            malformed_job = Path(root) / "job_worker_outputs"
            malformed_job.mkdir()
            with self.assertRaisesRegex(ValueError, "no job output directories"):
                Simulation.aggregate(root)


if __name__ == "__main__":
    unittest.main()
