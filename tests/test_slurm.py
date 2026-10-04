import os
import shlex
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np

from rmtpy.ensembles import GOE
from rmtpy.simulations import (
    PartialWidthsStatisticsSimulation,
    ResonanceStatisticsSimulation,
    Simulation,
    SpectralStatisticsSimulation,
    TimeDelayStatisticsSimulation,
    TransmissionCoefficientsSimulation,
)
from slurm.cluster import (
    GENERIC_PROFILE,
    ClusterProfile,
    PartitionLimits,
    ProfileDefaults,
    SlurmConfig,
    load_cluster_profile,
    load_selected_profile,
    resolve_profile_path,
    wall_time_minutes,
)
from slurm.configuration import (
    ScientificSpec,
    build_scientific_spec,
    child_seed,
    infer_job_name,
    instantiate_simulation,
    parse_mapping,
)
from slurm.runtime import (
    finish_calibration,
    load_calibration,
    merge_production,
    prepare_workspace,
    run_calibration_rank,
    run_production_rank,
)
from slurm.script import render_job_script, write_job_script
from tests.support import FloatVector, archive_fields, manifest_section

ENSEMBLE_INPUT: dict[str, object] = {
    "name": "GOE",
    "N": 4,
    "max_degree": 0,
    "seed": 123,
}

SIMULATION_INPUTS: tuple[tuple[str, dict[str, object], type[Simulation]], ...] = (
    ("spectral-statistics", {}, SpectralStatisticsSimulation),
    ("resonance-statistics", {}, ResonanceStatisticsSimulation),
    (
        "partial-widths-statistics",
        {"width_indices": [[0, 0], [0]]},
        PartialWidthsStatisticsSimulation,
    ),
    (
        "time-delay-statistics",
        {"energies": [-0.25, 0.25]},
        TimeDelayStatisticsSimulation,
    ),
    (
        "transmission-coefficients",
        {"channel_indices": [0]},
        TransmissionCoefficientsSimulation,
    ),
)


def build_spec(
    simulation: str = "spectral-statistics",
    simulation_arguments: dict[str, object] | None = None,
    *,
    tasks: int = 2,
    directory: str | Path = "outputs",
) -> ScientificSpec:
    return build_scientific_spec(
        simulation=simulation,
        ensemble=ENSEMBLE_INPUT,
        compound=None,
        simulation_arguments=simulation_arguments or {},
        realizs_per_task=1,
        tasks=tasks,
        directory=directory,
        plot=False,
    )


class SlurmConfigurationTests(unittest.TestCase):
    def test_mapping_parser_accepts_json_and_safe_python_literals(self) -> None:
        self.assertEqual(
            parse_mapping('{"name": "GOE", "N": 4}', label="ensemble"),
            {"name": "GOE", "N": 4},
        )
        self.assertEqual(
            parse_mapping("{'name': 'SYK', 'q': 4, 'N': 8}", label="ensemble"),
            {"name": "SYK", "q": 4, "N": 8},
        )
        with self.assertRaisesRegex(ValueError, "JSON or Python-literal"):
            _ = parse_mapping("__import__('os').system('false')", label="ensemble")
        with self.assertRaisesRegex(TypeError, "mapping with string keys"):
            _ = parse_mapping("{1: 'GOE'}", label="ensemble")

    def test_aliases_are_normalized_and_all_families_are_validated(self) -> None:
        for simulation_name, arguments, simulation_cls in SIMULATION_INPUTS:
            with self.subTest(simulation=simulation_name):
                spec = build_spec(simulation_name, arguments)
                simulation = instantiate_simulation(spec)
                assert isinstance(
                    simulation,
                    SpectralStatisticsSimulation
                    | ResonanceStatisticsSimulation
                    | PartialWidthsStatisticsSimulation
                    | TimeDelayStatisticsSimulation
                    | TransmissionCoefficientsSimulation,
                )
                self.assertIsInstance(simulation, simulation_cls)
                if isinstance(simulation, SpectralStatisticsSimulation):
                    ensemble = simulation.ensemble
                else:
                    ensemble = simulation.compound.ensemble
                self.assertIsInstance(ensemble, GOE)
                self.assertEqual(ensemble.num_majoranas, 4)
                self.assertEqual(ensemble.max_spectral_polynomial_degree, 0)
                self.assertEqual(simulation.realizs, 2)

        compound_spec = build_scientific_spec(
            simulation="resonance-statistics",
            ensemble=ENSEMBLE_INPUT,
            compound={"Nf": 1, "v": [0.5, 0.75]},
            simulation_arguments={},
            realizs_per_task=1,
            tasks=1,
            directory="outputs",
            plot=False,
        )
        compound_simulation = instantiate_simulation(compound_spec)
        assert isinstance(compound_simulation, ResonanceStatisticsSimulation)
        np.testing.assert_array_equal(
            compound_simulation.compound.couplings,
            np.array([0.5, 0.75]),
        )

    def test_invalid_scientific_inputs_fail_during_generation(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing=.*width_indices"):
            _ = build_spec("partial-widths-statistics")
        with self.assertRaisesRegex(ValueError, "Width state index"):
            _ = build_spec(
                "partial-widths-statistics",
                {"width_indices": [[2, 0]]},
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            _ = build_spec("time-delay-statistics", {"energies": [0.0, 0.0]})
        with self.assertRaisesRegex(ValueError, "does not accept a compound"):
            _ = build_scientific_spec(
                simulation="spectral-statistics",
                ensemble=ENSEMBLE_INPUT,
                compound={"Nf": 1},
                simulation_arguments={},
                realizs_per_task=1,
                tasks=1,
                directory="outputs",
                plot=False,
            )

    def test_seed_children_are_distinct_and_reproducible(self) -> None:
        spec = build_spec()
        first = child_seed(spec, phase=1, rank=0).generate_state(8)
        repeated = child_seed(spec, phase=1, rank=0).generate_state(8)
        other_rank = child_seed(spec, phase=1, rank=1).generate_state(8)
        calibration = child_seed(spec, phase=0, rank=0).generate_state(8)

        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(first, other_rank))
        self.assertFalse(np.array_equal(first, calibration))
        self.assertEqual(ScientificSpec.from_json(spec.to_json()), spec)

    def test_missing_seed_is_materialized_in_the_normalized_specification(self) -> None:
        spec = build_scientific_spec(
            simulation="spectral-statistics",
            ensemble={"name": "GOE", "N": 4, "max_degree": 0},
            compound=None,
            simulation_arguments={},
            realizs_per_task=1,
            tasks=1,
            directory="outputs",
            plot=False,
        )
        parameters = cast(dict[str, object], spec.ensemble["parameters"])
        self.assertIsInstance(parameters, dict)
        self.assertIsInstance(spec.master_seed, int)
        self.assertEqual(parameters["seed"], spec.master_seed)

    def test_job_name_is_inferred_from_common_aliases(self) -> None:
        self.assertEqual(
            infer_job_name({"name": "SYK", "q": 4, "N": 32}),
            "syk4_32",
        )
        self.assertEqual(
            infer_job_name({"name": "GOE", "num_majoranas": 12}),
            "goe_12",
        )


class SlurmProfileTests(unittest.TestCase):
    def test_generic_defaults_are_small_and_cluster_neutral(self) -> None:
        config = SlurmConfig()
        self.assertEqual(config.tasks, 1)
        self.assertEqual(config.nodes, 1)
        self.assertEqual(config.tasks_per_node, 1)
        self.assertEqual(config.cpus_per_task, 1)
        self.assertEqual(wall_time_minutes(config.wall_time), 60)
        self.assertIsNone(config.partition)
        self.assertIsNone(config.mail_type)
        self.assertEqual(config.modules, ())
        self.assertIsNone(config.conda_env)
        self.assertIsNone(config.cpu_bind)
        self.assertFalse(config.module_purge)
        self.assertFalse(config.use_numactl)

    def test_profile_loading_and_override_precedence(self) -> None:
        profile_source = """schema_version = 1
name = "test-cluster"

[defaults]
nodes = 2
tasks_per_node = 8
cpus_per_task = 4
time = "02:00:00"
partition = "compute"
mail_type = "END,FAIL"
modules = ["python", "blas"]
module_purge = true
conda_env = "rmtpy-env"
cpu_bind = "cores"
numa_policy = "2-3"

[partitions.compute]
cores_per_node = 32
max_nodes = 4
max_time = "04:00:00"
requires_memory = false
numactl = true
"""
        with tempfile.TemporaryDirectory() as root:
            profile_path = Path(root) / "cluster.toml"
            profile_path.write_text(profile_source, encoding="utf-8")
            profile = load_cluster_profile(profile_path)

            configured = profile.create_config(job_name="profiled")
            self.assertEqual(configured.nodes, 2)
            self.assertEqual(configured.tasks, 16)
            self.assertEqual(configured.partition, "compute")
            self.assertEqual(configured.modules, ("python", "blas"))
            self.assertTrue(configured.module_purge)
            self.assertEqual(configured.conda_env, "rmtpy-env")
            self.assertEqual(configured.cpu_bind, "cores")
            self.assertTrue(configured.use_numactl)
            self.assertEqual(configured.numa_policy, "2-3")

            overridden = profile.create_config(
                job_name="overridden",
                nodes=1,
                modules=[],
                module_purge=False,
                clear_conda=True,
                clear_cpu_bind=True,
                use_numactl=False,
            )
            self.assertEqual(overridden.nodes, 1)
            self.assertEqual(overridden.modules, ())
            self.assertFalse(overridden.module_purge)
            self.assertIsNone(overridden.conda_env)
            self.assertIsNone(overridden.cpu_bind)
            self.assertFalse(overridden.use_numactl)

            selected = load_selected_profile(
                None,
                environment={"RMTPY_SLURM_PROFILE_FILE": str(profile_path)},
            )
            self.assertEqual(selected, profile)
            self.assertEqual(
                resolve_profile_path(
                    "explicit.toml",
                    environment={"RMTPY_SLURM_PROFILE_FILE": str(profile_path)},
                ),
                Path("explicit.toml"),
            )
            self.assertEqual(load_selected_profile(None, environment={}), GENERIC_PROFILE)

    def test_invalid_profiles_fail_early(self) -> None:
        invalid_profiles = (
            (
                'schema_version = true\nname = "test"\n',
                "must be an integer",
            ),
            (
                'schema_version = 2\nname = "test"\n',
                "schema version",
            ),
            (
                'schema_version = 1\nname = "test"\nunknown = true\n',
                "Unknown profile key",
            ),
            (
                'schema_version = 1\nname = "test"\n'
                '[defaults]\nmail_user = "user@example.com"\n',
                "may not contain `mail_user`",
            ),
            (
                'schema_version = 1\nname = "test"\n[defaults]\nnodes = "many"\n',
                "positive integer",
            ),
            (
                'schema_version = 1\nname = "test"\n'
                '[defaults]\npartition = "missing"\n'
                "[partitions.compute]\nmax_nodes = 1\n",
                "is not defined",
            ),
            (
                'schema_version = 1\nname = "test"\n'
                '[partitions."bad partition"]\nmax_nodes = 1\n',
                "Partition name contains unsupported",
            ),
        )
        with tempfile.TemporaryDirectory() as root:
            profile_path = Path(root) / "profile.toml"
            for source, message in invalid_profiles:
                with self.subTest(message=message):
                    profile_path.write_text(source, encoding="utf-8")
                    with self.assertRaisesRegex((TypeError, ValueError), message):
                        _ = load_cluster_profile(profile_path)

            profile_path.write_text("not = [valid", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "valid TOML"):
                _ = load_cluster_profile(profile_path)
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                _ = load_cluster_profile(Path(root) / "missing.toml")

    def test_partition_limits_and_generic_partitions(self) -> None:
        profile = ClusterProfile(
            name="limited-cluster",
            defaults=ProfileDefaults(
                nodes=2,
                tasks_per_node=8,
                cpus_per_task=4,
                wall_time="02:00:00",
                partition="compute",
            ),
            partitions={
                "compute": PartitionLimits(
                    cores_per_node=32,
                    max_nodes=4,
                    max_minutes=4 * 60,
                ),
                "shared": PartitionLimits(
                    cores_per_node=32,
                    max_nodes=2,
                    max_minutes=2 * 60,
                    requires_memory=True,
                ),
            },
        )
        invalid_configurations = (
            ({"nodes": 5}, "at most 4 nodes"),
            ({"wall_time": "04:00:01"}, "at most 240 minutes"),
            ({"tasks_per_node": 9}, "exceeds.*32 cores"),
            ({"partition": "shared"}, "requires an explicit `--memory`"),
            ({"partition": "unknown"}, "is not defined"),
        )
        for keywords, message in invalid_configurations:
            with (
                self.subTest(keywords=keywords),
                self.assertRaisesRegex(ValueError, message),
            ):
                _ = profile.create_config(job_name="invalid", **keywords)

        shared = profile.create_config(
            job_name="shared",
            partition="shared",
            nodes=2,
            wall_time="02:00:00",
            memory="64G",
        )
        self.assertEqual(shared.memory, "64G")
        arbitrary = GENERIC_PROFILE.create_config(
            job_name="portable",
            partition="site-specific",
        )
        self.assertEqual(arbitrary.partition, "site-specific")

    def test_generic_script_snapshot_and_file_protection(self) -> None:
        spec = build_scientific_spec(
            simulation="spectral-statistics",
            ensemble={"name": "SYK", "q": 4, "N": 32, "seed": 123},
            compound=None,
            simulation_arguments={},
            realizs_per_task=4,
            tasks=1,
            directory="outputs/syk4_32",
            plot=False,
        )
        config = SlurmConfig(job_name="syk4_32")
        spec_argument = shlex.quote(spec.to_json())
        expected = f"""#!/bin/bash
#SBATCH --job-name=syk4_32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --distribution=cyclic
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_DYNAMIC=FALSE
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

cd "$SLURM_SUBMIT_DIR"

work_root=outputs/syk4_32/.rmtpy-slurm
work_dir="${{work_root}}/${{SLURM_JOB_ID}}"

python -m slurm.runtime prepare \\
  --work-dir "$work_dir" \\
  --spec-json {spec_argument}

srun python -m slurm.runtime calibrate \\
  --work-dir "$work_dir"

python -m slurm.runtime finish-calibration \\
  --work-dir "$work_dir"

srun python -m slurm.runtime worker \\
  --work-dir "$work_dir"

python -m slurm.runtime merge \\
  --work-dir "$work_dir"
"""
        script = render_job_script(spec, config)
        self.assertEqual(script, expected)
        self.assertNotIn("--mail-user", script)
        self.assertNotIn("@", script)
        self.assertNotIn("module ", script)
        self.assertNotIn("conda ", script)
        self.assertNotIn("numactl", script)
        self.assertNotIn("--partition", script)

        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            destination = root_path / "jobs" / "syk4_32.slurm"
            log_directory = root_path / "logs"
            _ = write_job_script(
                destination,
                script,
                log_directory=log_directory,
            )
            self.assertTrue(log_directory.is_dir())
            self.assertTrue(destination.stat().st_mode & stat.S_IXUSR)
            _ = subprocess.run(["bash", "-n", destination], check=True)

            with self.assertRaisesRegex(FileExistsError, "--force"):
                _ = write_job_script(
                    destination,
                    script,
                    log_directory=log_directory,
                )
            _ = write_job_script(
                destination,
                script + "# replaced\n",
                log_directory=log_directory,
                force=True,
            )
            self.assertTrue(
                destination.read_text(encoding="utf-8").endswith("replaced\n")
            )

    def test_optional_environment_lines_and_shell_values(self) -> None:
        spec = build_spec(directory="outputs/it's safe")
        config = SlurmConfig(
            job_name="configured",
            partition="compute",
            memory="64G",
            mail_type="BEGIN,END,FAIL",
            modules=("python", "blas"),
            module_purge=True,
            conda_env="rmtpy-env",
            cpu_bind="cores",
            use_numactl=True,
            numa_policy="2-3",
        )
        script = render_job_script(spec, config)
        self.assertIn("work_root='outputs/it'\"'\"'s safe/.rmtpy-slurm'", script)
        self.assertIn("#SBATCH --partition=compute", script)
        self.assertIn("#SBATCH --mem=64G", script)
        self.assertIn("#SBATCH --mail-type=BEGIN,END,FAIL", script)
        self.assertNotIn("#SBATCH --mail-user", script)
        self.assertIn("module purge", script)
        self.assertIn("module load python", script)
        self.assertIn("module load blas", script)
        self.assertIn("conda activate rmtpy-env", script)
        self.assertIn("conda deactivate", script)
        self.assertIn(
            "srun --cpu-bind=cores numactl --preferred-many=2-3",
            script,
        )
        _ = subprocess.run(["bash", "-n"], input=script, text=True, check=True)

    def test_public_example_profile_is_valid(self) -> None:
        profile = load_cluster_profile(
            Path(__file__).parents[1] / "slurm" / "profile.example.toml"
        )
        self.assertEqual(profile.name, "example-cluster")
        self.assertEqual(profile.defaults.partition, "compute")
        self.assertTrue(profile.partitions["compute-shared"].requires_memory)


class SlurmRuntimeTests(unittest.TestCase):
    def test_density_calibration_is_distributed_and_pooled_once(self) -> None:
        with (
            tempfile.TemporaryDirectory() as root,
            patch("rmtpy.density.NUM_HISTOGRAM_COUNTS", 2),
            patch("rmtpy.density.NUM_REALIZATIONS_MIN", 2),
        ):
            work_directory = Path(root) / "work"
            spec = build_scientific_spec(
                simulation="spectral-statistics",
                ensemble={"name": "GOE", "N": 4, "max_degree": 1, "seed": 123},
                compound=None,
                simulation_arguments={},
                realizs_per_task=1,
                tasks=2,
                directory=Path(root) / "outputs",
                plot=False,
            )
            _ = prepare_workspace(work_directory, spec)
            rank_sums: list[FloatVector] = []
            for rank in range(spec.tasks):
                with patch.dict(
                    os.environ,
                    {"SLURM_PROCID": str(rank), "SLURM_NTASKS": "2"},
                ):
                    contribution = run_calibration_rank(work_directory)
                fields = archive_fields(contribution)
                rank_sums.append(cast(FloatVector, fields["coefficient_sum"].copy()))
                self.assertEqual(cast(int, fields["sample_count"].item()), 1)

            _ = finish_calibration(work_directory)
            coefficients, sample_count = load_calibration(work_directory)
            self.assertEqual(sample_count, 2)
            assert coefficients is not None
            np.testing.assert_allclose(
                coefficients,
                (rank_sums[0] + rank_sums[1]) / 2,
            )

    def test_all_five_families_run_merge_round_trip_and_clean_up(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            for index, (simulation_name, arguments, simulation_cls) in enumerate(
                SIMULATION_INPUTS
            ):
                with self.subTest(simulation=simulation_name):
                    work_directory = root_path / f"work-{index}"
                    output_directory = root_path / f"outputs-{index}"
                    spec = build_spec(
                        simulation_name,
                        arguments,
                        directory=output_directory,
                    )
                    _ = prepare_workspace(work_directory, spec)
                    for rank in range(spec.tasks):
                        slurm_environment = {
                            "SLURM_PROCID": str(rank),
                            "SLURM_NTASKS": str(spec.tasks),
                        }
                        with patch.dict(os.environ, slurm_environment):
                            _ = run_calibration_rank(work_directory)
                    _ = finish_calibration(work_directory)
                    for rank in range(spec.tasks):
                        slurm_environment = {
                            "SLURM_PROCID": str(rank),
                            "SLURM_NTASKS": str(spec.tasks),
                        }
                        with patch.dict(os.environ, slurm_environment):
                            _ = run_production_rank(work_directory)

                    with patch.dict(os.environ, {"SLURM_JOB_ID": "unit-42"}):
                        destination = merge_production(work_directory)

                    restored = Simulation.load(destination)
                    assert isinstance(
                        restored,
                        SpectralStatisticsSimulation
                        | ResonanceStatisticsSimulation
                        | PartialWidthsStatisticsSimulation
                        | TimeDelayStatisticsSimulation
                        | TransmissionCoefficientsSimulation,
                    )
                    distributed = manifest_section(
                        restored.manifest.execution, "distributed"
                    )
                    self.assertIsInstance(restored, simulation_cls)
                    self.assertEqual(restored.realizs, 2)
                    self.assertEqual(distributed["job_id"], "unit-42")
                    self.assertEqual(distributed["tasks"], 2)
                    self.assertEqual(distributed["realizs_per_task"], 1)
                    self.assertEqual(distributed["total_realizs"], 2)
                    self.assertEqual(distributed["calibration_samples"], 0)
                    self.assertFalse(work_directory.exists())

    def test_failed_merge_retains_the_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            work_directory = Path(root) / "work"
            spec = build_spec(tasks=1, directory=Path(root) / "outputs")
            _ = prepare_workspace(work_directory, spec)
            with patch.dict(
                os.environ,
                {"SLURM_PROCID": "0", "SLURM_NTASKS": "1"},
            ):
                _ = run_calibration_rank(work_directory)
            _ = finish_calibration(work_directory)

            with self.assertRaisesRegex(ValueError, "did not complete"):
                _ = merge_production(work_directory)
            self.assertTrue(work_directory.is_dir())


if __name__ == "__main__":
    _ = unittest.main()
