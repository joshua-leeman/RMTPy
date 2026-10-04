import os
import shlex
from pathlib import Path

from .cluster import SlurmConfig
from .configuration import ScientificSpec


def _shell_path(path: str | Path, /) -> str:
    value = str(path)
    if "\n" in value or "\r" in value or "\0" in value:
        raise ValueError("Paths may not contain newlines or NUL characters.")
    return shlex.quote(value)


def render_job_script(
    spec: ScientificSpec,
    config: SlurmConfig,
    /,
) -> str:
    log_root = config.log_directory.as_posix().rstrip("/")
    if not log_root:
        raise ValueError("Log directory cannot be empty.")
    if any(character.isspace() for character in log_root):
        raise ValueError("Slurm log directory may not contain whitespace.")

    directives: list[str] = [
        f"#SBATCH --job-name={config.job_name}",
        f"#SBATCH --nodes={config.nodes}",
        f"#SBATCH --ntasks={config.tasks}",
        f"#SBATCH --ntasks-per-node={config.tasks_per_node}",
        f"#SBATCH --distribution={config.distribution}",
        f"#SBATCH --cpus-per-task={config.cpus_per_task}",
        f"#SBATCH --time={config.wall_time}",
    ]
    if config.partition is not None:
        directives.append(f"#SBATCH --partition={config.partition}")
    if config.memory is not None:
        directives.append(f"#SBATCH --mem={config.memory}")
    if config.mail_type is not None:
        directives.append(f"#SBATCH --mail-type={config.mail_type}")
    directives.extend(
        (
            f"#SBATCH --output={log_root}/%x_%j.out",
            f"#SBATCH --error={log_root}/%x_%j.err",
        )
    )

    environment_lines: list[str] = []
    if config.module_purge:
        environment_lines.append("module purge")
    environment_lines.extend(f"module load {module}" for module in config.modules)
    if config.conda_env is not None:
        environment_lines.append(f"conda activate {shlex.quote(config.conda_env)}")
    work_root = Path(spec.directory) / ".rmtpy-slurm"
    spec_argument = shlex.quote(spec.to_json())

    worker_prefix = ["srun"]
    if config.cpu_bind is not None:
        worker_prefix.append(f"--cpu-bind={config.cpu_bind}")
    if config.use_numactl:
        assert config.numa_policy is not None
        worker_prefix.extend(("numactl", f"--preferred-many={config.numa_policy}"))
    worker_command = " ".join(worker_prefix)

    lines = [
        "#!/bin/bash",
        *directives,
        "",
        "set -euo pipefail",
        "",
        "export OPENBLAS_NUM_THREADS=1",
        f"export OMP_NUM_THREADS={config.cpus_per_task}",
        f"export MKL_NUM_THREADS={config.cpus_per_task}",
        "export OPENBLAS_DYNAMIC=FALSE",
        "export OMP_DYNAMIC=FALSE",
        "export MKL_DYNAMIC=FALSE",
        "export PYTHONUNBUFFERED=1",
        "export MPLBACKEND=Agg",
        "",
        *environment_lines,
        *(("",) if environment_lines else ()),
        'cd "$SLURM_SUBMIT_DIR"',
        "",
        f"work_root={_shell_path(work_root)}",
        'work_dir="${work_root}/${SLURM_JOB_ID}"',
        "",
        "python -m slurm.runtime prepare \\",
        '  --work-dir "$work_dir" \\',
        f"  --spec-json {spec_argument}",
        "",
        f"{worker_command} python -m slurm.runtime calibrate \\",
        '  --work-dir "$work_dir"',
        "",
        "python -m slurm.runtime finish-calibration \\",
        '  --work-dir "$work_dir"',
        "",
        f"{worker_command} python -m slurm.runtime worker \\",
        '  --work-dir "$work_dir"',
        "",
        "python -m slurm.runtime merge \\",
        '  --work-dir "$work_dir"',
    ]
    if config.conda_env is not None:
        lines.extend(("", "conda deactivate"))
    lines.append("")
    return "\n".join(lines)


def write_job_script(
    path: str | Path,
    script: str,
    /,
    *,
    log_directory: str | Path,
    force: bool = False,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Path(log_directory).mkdir(parents=True, exist_ok=True)

    mode = "w" if force else "x"
    try:
        with destination.open(mode, encoding="utf-8", newline="\n") as file:
            _ = file.write(script)
            file.flush()
            os.fsync(file.fileno())
    except FileExistsError as exc:
        raise FileExistsError(
            f"`{destination}` already exists; pass `--force` to replace it."
        ) from exc

    destination.chmod(destination.stat().st_mode | 0o100)
    return destination
