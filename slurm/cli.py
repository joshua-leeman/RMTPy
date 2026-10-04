import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from .cluster import (
    GENERIC_PROFILE,
    PROFILE_ENVIRONMENT_VARIABLE,
    ClusterProfile,
    load_selected_profile,
)
from .configuration import (
    build_scientific_spec,
    infer_job_name,
    parse_mapping,
)
from .script import render_job_script, write_job_script


class SlurmArguments(argparse.Namespace):
    profile_file: Path | None = None
    simulation: str = ""
    ensemble: str = ""
    compound: str | None = None
    simulation_args: str = "{}"
    realizs_per_task: int = 0
    directory: str = "outputs"
    job_name: str | None = None
    output: Path | None = None
    no_plot: bool = False
    force: bool = False
    nodes: int | None = None
    tasks_per_node: int | None = None
    cpus_per_task: int | None = None
    wall_time: str | None = None
    partition: str | None = None
    clear_partition: bool = False
    memory: str | None = None
    distribution: str | None = None
    mail_type: str | None = None
    conda_env: str | None = None
    clear_conda: bool = False
    log_directory: Path | None = None
    modules: list[str] | None = None
    module_purge: bool | None = None
    cpu_bind: str | None = None
    clear_cpu_bind: bool = False
    numactl: bool | None = None
    numa_policy: str | None = None


def _profile_selector() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    _ = parser.add_argument("--profile-file", type=Path)
    return parser


def build_parser(profile: ClusterProfile = GENERIC_PROFILE) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m slurm",
        description=(
            "Validate an RMTPy simulation and write a distributed Slurm job "
            f"using the {profile.name!r} cluster profile."
        ),
    )
    _ = parser.add_argument(
        "--profile-file",
        type=Path,
        help=(
            f"Private TOML cluster profile. Overrides ${PROFILE_ENVIRONMENT_VARIABLE}."
        ),
    )
    _ = parser.add_argument("--simulation", "-sim", required=True)
    _ = parser.add_argument(
        "--ensemble",
        "-ens",
        required=True,
        help="JSON or Python mapping, e.g. \"{'name':'SYK','q':4,'N':32}\".",
    )
    _ = parser.add_argument(
        "--compound",
        "-comp",
        help="Optional compound mapping; its type is inferred when `name` is omitted.",
    )
    _ = parser.add_argument(
        "--simulation-args",
        default="{}",
        help="Family-specific mapping such as energies or width/channel indices.",
    )
    _ = parser.add_argument(
        "--realizs-per-task",
        "-r",
        type=int,
        required=True,
        help="Monte Carlo realizations executed by every Slurm task.",
    )
    _ = parser.add_argument("--directory", "-dir", default="outputs")
    _ = parser.add_argument("--job-name")
    _ = parser.add_argument(
        "--output",
        type=Path,
        help="Submission script path; defaults to <job-name>.slurm.",
    )
    _ = parser.add_argument("--no-plot", action="store_true")
    _ = parser.add_argument("--force", action="store_true")

    resources = parser.add_argument_group("Slurm resources")
    _ = resources.add_argument("--nodes", type=int)
    _ = resources.add_argument(
        "--ntasks-per-node",
        "--tasks-per-node",
        dest="tasks_per_node",
        type=int,
    )
    _ = resources.add_argument("--cpus-per-task", type=int)
    _ = resources.add_argument("--time", dest="wall_time")
    partition_group = resources.add_mutually_exclusive_group()
    _ = partition_group.add_argument("--partition")
    _ = partition_group.add_argument(
        "--no-partition",
        action="store_true",
        dest="clear_partition",
        help="Omit a partition supplied by the selected profile.",
    )
    _ = resources.add_argument("--memory")
    _ = resources.add_argument("--distribution", choices=("block", "cyclic"))

    environment = parser.add_argument_group("Slurm environment")
    _ = environment.add_argument(
        "--mail-type",
        help=(
            "Notification events only. Pass a private mail address directly to "
            "sbatch; generated scripts never contain --mail-user."
        ),
    )
    conda_group = environment.add_mutually_exclusive_group()
    _ = conda_group.add_argument("--conda-env")
    _ = conda_group.add_argument(
        "--no-conda",
        action="store_true",
        dest="clear_conda",
        help="Disable conda activation supplied by the selected profile.",
    )
    _ = environment.add_argument("--log-directory", type=Path)
    module_group = environment.add_mutually_exclusive_group()
    _ = module_group.add_argument(
        "--module",
        dest="modules",
        action="append",
        help="Replace profile modules; repeat for multiple modules.",
    )
    _ = module_group.add_argument(
        "--no-modules",
        dest="modules",
        action="store_const",
        const=[],
        help="Clear modules supplied by the selected profile.",
    )
    _ = environment.add_argument(
        "--module-purge",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Purge inherited modules before optional module loads.",
    )
    binding_group = environment.add_mutually_exclusive_group()
    _ = binding_group.add_argument("--cpu-bind")
    _ = binding_group.add_argument(
        "--no-cpu-bind",
        action="store_true",
        dest="clear_cpu_bind",
        help="Disable srun CPU binding supplied by the selected profile.",
    )
    _ = environment.add_argument(
        "--numactl",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run workers through numactl using --numa-policy.",
    )
    _ = environment.add_argument("--numa-policy")
    return parser


def generate_from_arguments(
    arguments: argparse.Namespace,
    profile: ClusterProfile = GENERIC_PROFILE,
) -> Path:
    arguments = cast(SlurmArguments, arguments)
    ensemble_mapping = parse_mapping(arguments.ensemble, label="Ensemble input")
    compound_mapping = (
        None
        if arguments.compound is None
        else parse_mapping(arguments.compound, label="Compound input")
    )
    simulation_arguments = parse_mapping(
        arguments.simulation_args,
        label="Simulation arguments",
    )
    job_name = arguments.job_name or infer_job_name(ensemble_mapping)

    cluster = profile.create_config(
        job_name=job_name,
        nodes=arguments.nodes,
        tasks_per_node=arguments.tasks_per_node,
        cpus_per_task=arguments.cpus_per_task,
        wall_time=arguments.wall_time,
        partition=arguments.partition,
        clear_partition=arguments.clear_partition,
        memory=arguments.memory,
        distribution=arguments.distribution,
        mail_type=arguments.mail_type,
        conda_env=arguments.conda_env,
        clear_conda=arguments.clear_conda,
        log_directory=arguments.log_directory,
        modules=arguments.modules,
        module_purge=arguments.module_purge,
        cpu_bind=arguments.cpu_bind,
        clear_cpu_bind=arguments.clear_cpu_bind,
        use_numactl=arguments.numactl,
        numa_policy=arguments.numa_policy,
    )
    scientific = build_scientific_spec(
        simulation=arguments.simulation,
        ensemble=ensemble_mapping,
        compound=compound_mapping,
        simulation_arguments=simulation_arguments,
        realizs_per_task=arguments.realizs_per_task,
        tasks=cluster.tasks,
        directory=arguments.directory,
        plot=not arguments.no_plot,
    )

    destination = arguments.output or Path(f"{job_name}.slurm")
    script = render_job_script(scientific, cluster)
    return write_job_script(
        destination,
        script,
        log_directory=cluster.log_directory,
        force=arguments.force,
    )


def main(argv: Sequence[str] | None = None) -> int:
    selector = _profile_selector()
    selected, _ = selector.parse_known_args(argv)
    selected = cast(SlurmArguments, selected)
    try:
        profile = load_selected_profile(selected.profile_file)
    except (OSError, TypeError, ValueError) as exc:
        build_parser().error(str(exc))

    parser = build_parser(profile)
    arguments = parser.parse_args(argv, namespace=SlurmArguments())
    try:
        destination = generate_from_arguments(arguments, profile)
    except (KeyError, TypeError, ValueError, FileExistsError) as exc:
        parser.error(str(exc))

    print(destination)
    return 0
