import argparse
import json
import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np

from rmtpy.conversion import json_value
from rmtpy.simulations import (
    ResonanceStatisticsSimulation,
    Simulation,
    SpectralStatisticsSimulation,
    TimeDelayStatisticsSimulation,
)
from rmtpy.simulations.base_simulation import ExecutionState

from .configuration import ScientificSpec, child_seed, instantiate_simulation

SPEC_FILE_NAME: str = "spec.json"
WORKSPACE_MARKER: str = ".rmtpy-slurm-workspace"
CALIBRATION_DIRECTORY_NAME: str = "calibration"
CALIBRATION_FILE_NAME: str = "calibration.npz"
PRODUCTION_DIRECTORY_NAME: str = "production"
COMPLETE_MARKER: str = "complete"


def _write_json(path: Path, value: object, /) -> None:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("x", encoding="utf-8") as file:
        _ = file.write(encoded)
        _ = file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    _ = temporary_path.replace(path)


def _write_npz(
    path: Path,
    /,
    **arrays: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("xb") as file:
        np.savez_compressed(file, allow_pickle=False, **arrays)
        file.flush()
        os.fsync(file.fileno())
    _ = temporary_path.replace(path)


def prepare_workspace(work_directory: str | Path, spec: ScientificSpec, /) -> Path:
    work_directory = Path(work_directory)
    if work_directory.exists():
        raise FileExistsError(f"Workspace `{work_directory}` already exists.")
    work_directory.mkdir(parents=True)
    (work_directory / WORKSPACE_MARKER).touch(exist_ok=False)
    _write_json(work_directory / SPEC_FILE_NAME, spec.to_dict())
    (work_directory / CALIBRATION_DIRECTORY_NAME).mkdir()
    (work_directory / PRODUCTION_DIRECTORY_NAME).mkdir()
    return work_directory


def load_spec(work_directory: str | Path, /) -> ScientificSpec:
    work_directory = Path(work_directory)
    if not (work_directory / WORKSPACE_MARKER).is_file():
        raise ValueError(f"`{work_directory}` is not an RMTPy Slurm workspace.")
    value = cast(
        object, json.loads((work_directory / SPEC_FILE_NAME).read_text(encoding="utf-8"))
    )
    if not isinstance(value, dict):
        raise TypeError("Workspace scientific specification is malformed.")
    return ScientificSpec.from_dict(cast(dict[str, object], value))


def _slurm_rank_and_size(spec: ScientificSpec) -> tuple[int, int]:
    try:
        rank = int(os.environ["SLURM_PROCID"])
        tasks = int(os.environ["SLURM_NTASKS"])
    except (KeyError, ValueError) as exc:
        raise RuntimeError(
            "Distributed runtime commands must be launched by `srun`."
        ) from exc
    if tasks != spec.tasks:
        raise ValueError(f"Slurm launched {tasks} tasks; the job expects {spec.tasks}.")
    if not 0 <= rank < tasks:
        raise ValueError(f"Invalid Slurm process rank {rank} for {tasks} tasks.")
    return rank, tasks


def _calibration_density(simulation: Simulation):
    if isinstance(simulation, SpectralStatisticsSimulation):
        return simulation.ensemble.spectral_density
    if isinstance(simulation, ResonanceStatisticsSimulation):
        return simulation.compound.resonance_density
    if isinstance(simulation, TimeDelayStatisticsSimulation):
        return simulation.compound.ensemble.spectral_density
    return None


def run_calibration_rank(work_directory: str | Path, /) -> Path:
    work_directory = Path(work_directory)
    spec = load_spec(work_directory)
    rank, tasks = _slurm_rank_and_size(spec)
    simulation = instantiate_simulation(
        spec,
        realizs=1,
        seed=child_seed(spec, phase=0, rank=rank),
    )
    density = _calibration_density(simulation)

    if density is None or density.max_polynomial_degree == 0:
        coefficient_sum = np.empty(0, dtype=np.float64)
        sample_count = 0
    else:
        sample_count = len(range(rank, density.optimal_realizs, tasks))
        coefficient_sum = np.zeros(
            density.max_polynomial_degree + 1,
            dtype=np.float64,
        )
        for sample in density.sample_stream(realizs=sample_count):
            coefficient_sum += density.compute_variate_coeffs(sample)

    destination = work_directory / CALIBRATION_DIRECTORY_NAME / f"rank_{rank:06d}.npz"
    _write_npz(
        destination,
        coefficient_sum=coefficient_sum,
        sample_count=np.array(sample_count, dtype=np.int64),
    )
    return destination


def finish_calibration(work_directory: str | Path, /) -> Path:
    work_directory = Path(work_directory)
    spec = load_spec(work_directory)
    coefficient_sum: np.ndarray[tuple[int], np.dtype[np.floating]] | None = None
    sample_count = 0

    for rank in range(spec.tasks):
        path = work_directory / CALIBRATION_DIRECTORY_NAME / f"rank_{rank:06d}.npz"
        if not path.is_file():
            raise ValueError(f"Calibration contribution `{path}` is missing.")
        with cast(np.lib.npyio.NpzFile, np.load(path, allow_pickle=False)) as archive:
            rank_sum = np.asarray(archive["coefficient_sum"], dtype=np.float64)
            rank_count = cast(int, archive["sample_count"].item())
        if rank_count < 0:
            raise ValueError(f"Calibration contribution `{path}` has a negative count.")
        if coefficient_sum is None:
            coefficient_sum = np.zeros_like(rank_sum)
        if rank_sum.shape != coefficient_sum.shape or not np.all(np.isfinite(rank_sum)):
            raise ValueError(f"Calibration contribution `{path}` is malformed.")
        coefficient_sum += rank_sum
        sample_count += rank_count

    if coefficient_sum is None:
        raise ValueError("Calibration produced no contributions.")
    if coefficient_sum.size > 0:
        if sample_count <= 0:
            raise ValueError("Calibration has coefficients but no samples.")
        average_coefficients = coefficient_sum / sample_count
    else:
        average_coefficients = coefficient_sum

    destination = work_directory / CALIBRATION_FILE_NAME
    _write_npz(
        destination,
        average_coefficients=average_coefficients,
        sample_count=np.array(sample_count, dtype=np.int64),
    )
    return destination


def load_calibration(
    work_directory: str | Path,
    /,
) -> tuple[np.ndarray[tuple[int], np.dtype[np.floating]] | None, int]:
    path = Path(work_directory) / CALIBRATION_FILE_NAME
    if not path.is_file():
        raise ValueError(f"Final calibration `{path}` is missing.")
    with cast(np.lib.npyio.NpzFile, np.load(path, allow_pickle=False)) as archive:
        coefficients = np.asarray(archive["average_coefficients"], dtype=np.float64)
        sample_count = cast(int, archive["sample_count"].item())
    return (None if coefficients.size == 0 else coefficients), sample_count


def run_production_rank(work_directory: str | Path, /) -> Path:
    work_directory = Path(work_directory)
    spec = load_spec(work_directory)
    rank, _ = _slurm_rank_and_size(spec)
    calibration, _ = load_calibration(work_directory)
    simulation = instantiate_simulation(
        spec,
        realizs=spec.realizs_per_task,
        seed=child_seed(spec, phase=1, rank=rank),
        calibration=calibration,
    )
    simulation.execute()

    rank_directory = work_directory / PRODUCTION_DIRECTORY_NAME / f"job_{rank}_outputs"
    rank_directory.mkdir()
    destination = simulation.save(rank_directory)
    (rank_directory / COMPLETE_MARKER).touch(exist_ok=False)
    return destination


def _replace_manifest_configuration(
    aggregate: Simulation,
    spec: ScientificSpec,
) -> None:
    reference = instantiate_simulation(spec, realizs=spec.total_realizs)
    aggregate.manifest.configuration.clear()
    aggregate.manifest.configuration.update(reference.manifest.configuration)


def _record_distributed_execution(
    aggregate: Simulation,
    spec: ScientificSpec,
    *,
    calibration_samples: int,
) -> None:
    final_state = aggregate.manifest.rng.get("final_state")
    bit_generator = aggregate.manifest.rng["bit_generator"]
    aggregate.manifest.rng.clear()
    aggregate.manifest.rng.update(
        {
            "policy": "numpy.random.SeedSequence",
            "bit_generator": bit_generator,
            "seed": json_value(spec.master_seed),
            "state_policy": "deterministic_child_streams",
            "spawn_policy": {
                "calibration": "spawn_key=(0, rank)",
                "production": "spawn_key=(1, rank)",
            },
            "final_state": final_state,
        }
    )
    aggregate.manifest.execution["distributed"] = {
        "scheduler": "slurm",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "tasks": spec.tasks,
        "realizs_per_task": spec.realizs_per_task,
        "total_realizs": spec.total_realizs,
        "calibration_samples": calibration_samples,
        "workspace_policy": "remove_on_success_retain_on_failure",
    }


def _plot_aggregate(
    aggregate: Simulation,
    destination: Path,
    spec: ScientificSpec,
) -> None:
    if not spec.plot:
        return
    if isinstance(aggregate, TimeDelayStatisticsSimulation):
        spectral_directory = spec.plot_arguments.get("spectral_statistics_directory")
        aggregate.plot(
            destination,
            spectral_statistics_directory=(
                None if spectral_directory is None else str(spectral_directory)
            ),
        )
        return
    aggregate.plot(destination)


def _remove_workspace(work_directory: Path, /) -> None:
    marker = work_directory / WORKSPACE_MARKER
    if not marker.is_file():
        raise ValueError(f"Refusing to remove unmarked workspace `{work_directory}`.")
    shutil.rmtree(work_directory)


def merge_production(work_directory: str | Path, /) -> Path:
    work_directory = Path(work_directory)
    spec = load_spec(work_directory)
    _, calibration_samples = load_calibration(work_directory)
    production_directory = work_directory / PRODUCTION_DIRECTORY_NAME
    for rank in range(spec.tasks):
        marker = production_directory / f"job_{rank}_outputs" / COMPLETE_MARKER
        if not marker.is_file():
            raise ValueError(f"Production rank {rank} did not complete.")

    aggregate = Simulation.aggregate(production_directory)
    if aggregate.execution_state is not ExecutionState.COMPLETE:
        raise RuntimeError("Aggregated simulation is not complete.")
    _replace_manifest_configuration(aggregate, spec)
    _record_distributed_execution(
        aggregate,
        spec,
        calibration_samples=calibration_samples,
    )
    destination = aggregate.save(spec.directory)
    _plot_aggregate(aggregate, destination, spec)
    _remove_workspace(work_directory)
    return destination


class RuntimeArguments(argparse.Namespace):
    command: str = ""
    work_dir: Path = Path(".")
    spec_json: str = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m slurm.runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    _ = prepare.add_argument("--work-dir", type=Path, required=True)
    _ = prepare.add_argument("--spec-json", required=True)

    for command in ("calibrate", "finish-calibration", "worker", "merge"):
        subparser = subparsers.add_parser(command)
        _ = subparser.add_argument("--work-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv, namespace=RuntimeArguments())
    if arguments.command == "prepare":
        destination = prepare_workspace(
            arguments.work_dir,
            ScientificSpec.from_json(arguments.spec_json),
        )
    elif arguments.command == "calibrate":
        destination = run_calibration_rank(arguments.work_dir)
    elif arguments.command == "finish-calibration":
        destination = finish_calibration(arguments.work_dir)
    elif arguments.command == "worker":
        destination = run_production_rank(arguments.work_dir)
    elif arguments.command == "merge":
        destination = merge_production(arguments.work_dir)
    else:
        raise AssertionError(f"Unknown runtime command {arguments.command!r}.")

    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
