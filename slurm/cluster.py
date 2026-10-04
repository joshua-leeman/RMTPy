import os
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import attrs

PROFILE_ENVIRONMENT_VARIABLE = "RMTPY_SLURM_PROFILE_FILE"
PROFILE_SCHEMA_VERSION = 1

DEFAULT_JOB_NAME = "rmtpy"
DEFAULT_NODES = 1
DEFAULT_TASKS_PER_NODE = 1
DEFAULT_CPUS_PER_TASK = 1
DEFAULT_TIME = "01:00:00"
DEFAULT_DISTRIBUTION = "cyclic"
DEFAULT_LOG_DIRECTORY = Path("logs")

_PROFILE_KEYS = {"schema_version", "name", "defaults", "partitions"}
_DEFAULT_KEYS = {
    "nodes",
    "tasks_per_node",
    "cpus_per_task",
    "time",
    "partition",
    "memory",
    "distribution",
    "mail_type",
    "log_directory",
    "modules",
    "module_purge",
    "conda_env",
    "cpu_bind",
    "numactl",
    "numa_policy",
}
_PARTITION_KEYS = {
    "cores_per_node",
    "max_nodes",
    "max_time",
    "requires_memory",
    "numactl",
}
_MAIL_TYPES = {
    "NONE",
    "BEGIN",
    "END",
    "FAIL",
    "REQUEUE",
    "ALL",
    "INVALID_DEPEND",
    "STAGE_OUT",
    "TIME_LIMIT",
    "TIME_LIMIT_90",
    "TIME_LIMIT_80",
    "TIME_LIMIT_50",
    "ARRAY_TASKS",
}


def wall_time_minutes(value: str, /) -> int:
    match = re.fullmatch(r"(?:(\d+)-)?(\d{1,3}):(\d{2}):(\d{2})", value)
    if match is None:
        raise ValueError("Wall time must use `[days-]hours:minutes:seconds` format.")
    days, hours, minutes, seconds = (
        0 if match[1] is None else int(match[1]),
        int(match[2]),
        int(match[3]),
        int(match[4]),
    )
    if minutes >= 60 or seconds >= 60 or (days > 0 and hours >= 24):
        raise ValueError("Wall time contains an out-of-range component.")
    total_seconds = (((days * 24) + hours) * 60 + minutes) * 60 + seconds
    if total_seconds <= 0:
        raise ValueError("Wall time must be positive.")
    return (total_seconds + 59) // 60


def _validate_token(value: str, /, *, label: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.+/:-]+", value):
        raise ValueError(f"{label} contains unsupported shell or Slurm characters.")


def _positive_integer(value: object, /, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TypeError(f"{label} must be a positive integer.")
    return value


def _optional_string(value: object, /, *, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a nonempty string.")
    return value


def _boolean(value: object, /, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be boolean.")
    return value


def _mapping(value: object, /, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a TOML table.")
    return cast(dict[str, object], value)


def _reject_unknown_keys(
    mapping: Mapping[str, object],
    allowed: set[str],
    /,
    *,
    label: str,
) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        rendered = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown {label} key(s): {rendered}.")


def _reject_mail_user(value: object, /, *, path: str = "profile") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized == "mail_user":
                raise ValueError(
                    "Cluster profiles may not contain `mail_user`; pass a private "
                    "environment variable to `sbatch --mail-user` instead."
                )
            _reject_mail_user(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_mail_user(item, path=f"{path}[{index}]")


@attrs.frozen(kw_only=True, eq=True, weakref_slot=False)
class PartitionLimits:
    cores_per_node: int | None = None
    max_nodes: int | None = None
    max_minutes: int | None = None
    requires_memory: bool = False
    use_numactl: bool | None = None


@attrs.frozen(kw_only=True, eq=True, weakref_slot=False)
class ProfileDefaults:
    nodes: int = DEFAULT_NODES
    tasks_per_node: int = DEFAULT_TASKS_PER_NODE
    cpus_per_task: int = DEFAULT_CPUS_PER_TASK
    wall_time: str = DEFAULT_TIME
    partition: str | None = None
    memory: str | None = None
    distribution: str = DEFAULT_DISTRIBUTION
    mail_type: str | None = None
    log_directory: Path = attrs.field(
        default=DEFAULT_LOG_DIRECTORY,
        converter=Path,
    )
    modules: tuple[str, ...] = attrs.field(factory=tuple, converter=tuple)
    module_purge: bool = False
    conda_env: str | None = None
    cpu_bind: str | None = None
    use_numactl: bool = False
    numa_policy: str | None = None


@attrs.frozen(kw_only=True, eq=True, weakref_slot=False)
class ClusterProfile:
    name: str = "generic"
    defaults: ProfileDefaults = attrs.field(factory=ProfileDefaults)
    partitions: dict[str, PartitionLimits] = attrs.field(factory=dict, converter=dict)

    def create_config(
        self,
        *,
        job_name: str,
        nodes: int | None = None,
        tasks_per_node: int | None = None,
        cpus_per_task: int | None = None,
        wall_time: str | None = None,
        partition: str | None = None,
        clear_partition: bool = False,
        memory: str | None = None,
        distribution: str | None = None,
        mail_type: str | None = None,
        log_directory: Path | None = None,
        modules: list[str] | tuple[str, ...] | None = None,
        module_purge: bool | None = None,
        conda_env: str | None = None,
        clear_conda: bool = False,
        cpu_bind: str | None = None,
        clear_cpu_bind: bool = False,
        use_numactl: bool | None = None,
        numa_policy: str | None = None,
    ) -> SlurmConfig:
        selected_partition = (
            None
            if clear_partition
            else self.defaults.partition
            if partition is None
            else partition
        )
        limits = (
            None
            if selected_partition is None
            else self.partitions.get(selected_partition)
        )
        selected_numactl = use_numactl
        if selected_numactl is None:
            selected_numactl = (
                limits.use_numactl
                if limits is not None and limits.use_numactl is not None
                else self.defaults.use_numactl
            )

        return SlurmConfig(
            profile_name=self.name,
            partition_limits=self.partitions,
            job_name=job_name,
            nodes=self.defaults.nodes if nodes is None else nodes,
            tasks_per_node=(
                self.defaults.tasks_per_node if tasks_per_node is None else tasks_per_node
            ),
            cpus_per_task=(
                self.defaults.cpus_per_task if cpus_per_task is None else cpus_per_task
            ),
            wall_time=self.defaults.wall_time if wall_time is None else wall_time,
            partition=selected_partition,
            memory=self.defaults.memory if memory is None else memory,
            distribution=(
                self.defaults.distribution if distribution is None else distribution
            ),
            mail_type=(self.defaults.mail_type if mail_type is None else mail_type),
            log_directory=(
                self.defaults.log_directory if log_directory is None else log_directory
            ),
            modules=self.defaults.modules if modules is None else tuple(modules),
            module_purge=(
                self.defaults.module_purge if module_purge is None else module_purge
            ),
            conda_env=(
                None
                if clear_conda
                else self.defaults.conda_env
                if conda_env is None
                else conda_env
            ),
            cpu_bind=(
                None
                if clear_cpu_bind
                else self.defaults.cpu_bind
                if cpu_bind is None
                else cpu_bind
            ),
            use_numactl=selected_numactl,
            numa_policy=(
                self.defaults.numa_policy if numa_policy is None else numa_policy
            ),
        )


@attrs.frozen(kw_only=True, eq=True, weakref_slot=False)
class SlurmConfig:
    profile_name: str = "generic"
    partition_limits: dict[str, PartitionLimits] = attrs.field(
        factory=dict,
        converter=dict,
        repr=False,
    )
    job_name: str = DEFAULT_JOB_NAME
    nodes: int = DEFAULT_NODES
    tasks_per_node: int = DEFAULT_TASKS_PER_NODE
    cpus_per_task: int = DEFAULT_CPUS_PER_TASK
    wall_time: str = DEFAULT_TIME
    partition: str | None = None
    memory: str | None = None
    distribution: str = DEFAULT_DISTRIBUTION
    mail_type: str | None = None
    log_directory: Path = attrs.field(
        default=DEFAULT_LOG_DIRECTORY,
        converter=Path,
    )
    modules: tuple[str, ...] = attrs.field(factory=tuple, converter=tuple)
    module_purge: bool = False
    conda_env: str | None = None
    cpu_bind: str | None = None
    use_numactl: bool = False
    numa_policy: str | None = None

    def __attrs_post_init__(self) -> None:
        for label, value in (
            ("nodes", self.nodes),
            ("tasks_per_node", self.tasks_per_node),
            ("cpus_per_task", self.cpus_per_task),
        ):
            _ = _positive_integer(value, label=label)

        requested_minutes = wall_time_minutes(self.wall_time)
        limits: PartitionLimits | None = None
        if self.partition is not None:
            _validate_token(self.partition, label="Partition")
            if self.partition_limits:
                try:
                    limits = self.partition_limits[self.partition]
                except KeyError as exc:
                    raise ValueError(
                        f"Partition {self.partition!r} is not defined by profile "
                        f"{self.profile_name!r}."
                    ) from exc

        if limits is not None:
            if limits.max_nodes is not None and self.nodes > limits.max_nodes:
                raise ValueError(
                    f"{self.partition} permits at most {limits.max_nodes} nodes."
                )
            if (
                limits.cores_per_node is not None
                and self.tasks_per_node * self.cpus_per_task > limits.cores_per_node
            ):
                raise ValueError(
                    "tasks-per-node times CPUs-per-task exceeds the partition's "
                    + f"{limits.cores_per_node} cores per node."
                )
            if limits.max_minutes is not None and requested_minutes > limits.max_minutes:
                raise ValueError(
                    f"{self.partition} permits at most {limits.max_minutes} minutes."
                )
            if limits.requires_memory and self.memory is None:
                raise ValueError(
                    f"Partition {self.partition!r} requires an explicit `--memory`."
                )

        if self.memory is not None and not re.fullmatch(
            r"\d+(?:\.\d+)?[KMGTP]?[Bb]?", self.memory
        ):
            raise ValueError("Memory must look like `64G` or `128GB`.")
        if self.distribution not in {"block", "cyclic"}:
            raise ValueError("Distribution must be `block` or `cyclic`.")
        if self.mail_type is not None:
            mail_types = set(self.mail_type.split(","))
            if not mail_types or not mail_types <= _MAIL_TYPES:
                raise ValueError("Mail type contains an unsupported notification event.")
        if self.use_numactl and self.numa_policy is None:
            raise ValueError("NUMA preference requires a `--numa-policy` value.")

        _validate_token(self.profile_name, label="Profile name")
        _validate_token(self.job_name, label="Job name")
        if self.conda_env is not None:
            _validate_token(self.conda_env, label="Conda environment")
        if self.cpu_bind is not None:
            _validate_token(self.cpu_bind, label="CPU binding")
        if self.numa_policy is not None:
            _validate_token(self.numa_policy, label="NUMA policy")
        for module in self.modules:
            _validate_token(module, label="Module name")

    @property
    def tasks(self) -> int:
        return self.nodes * self.tasks_per_node

    @property
    def partition_details(self) -> PartitionLimits | None:
        if self.partition is None:
            return None
        return self.partition_limits.get(self.partition)


GENERIC_PROFILE = ClusterProfile()


def _parse_profile_defaults(value: object, /) -> ProfileDefaults:
    mapping = _mapping(value, label="Profile defaults")
    _reject_unknown_keys(mapping, _DEFAULT_KEYS, label="profile default")

    nodes = _positive_integer(mapping.get("nodes", DEFAULT_NODES), label="defaults.nodes")
    tasks_per_node = _positive_integer(
        mapping.get("tasks_per_node", DEFAULT_TASKS_PER_NODE),
        label="defaults.tasks_per_node",
    )
    cpus_per_task = _positive_integer(
        mapping.get("cpus_per_task", DEFAULT_CPUS_PER_TASK),
        label="defaults.cpus_per_task",
    )
    wall_time = _optional_string(mapping.get("time", DEFAULT_TIME), label="defaults.time")
    assert wall_time is not None
    _ = wall_time_minutes(wall_time)

    partition = _optional_string(mapping.get("partition"), label="defaults.partition")
    memory = _optional_string(mapping.get("memory"), label="defaults.memory")
    distribution = _optional_string(
        mapping.get("distribution", DEFAULT_DISTRIBUTION),
        label="defaults.distribution",
    )
    assert distribution is not None
    mail_type = _optional_string(mapping.get("mail_type"), label="defaults.mail_type")
    log_directory = _optional_string(
        mapping.get("log_directory", str(DEFAULT_LOG_DIRECTORY)),
        label="defaults.log_directory",
    )
    assert log_directory is not None

    modules_value = mapping.get("modules", [])
    if not isinstance(modules_value, list) or not all(
        isinstance(module, str) and module for module in modules_value
    ):
        raise TypeError("defaults.modules must be an array of nonempty strings.")
    modules = tuple(cast(list[str], modules_value))

    return ProfileDefaults(
        nodes=nodes,
        tasks_per_node=tasks_per_node,
        cpus_per_task=cpus_per_task,
        wall_time=wall_time,
        partition=partition,
        memory=memory,
        distribution=distribution,
        mail_type=mail_type,
        log_directory=Path(log_directory),
        modules=modules,
        module_purge=_boolean(
            mapping.get("module_purge", False), label="defaults.module_purge"
        ),
        conda_env=_optional_string(mapping.get("conda_env"), label="defaults.conda_env"),
        cpu_bind=_optional_string(mapping.get("cpu_bind"), label="defaults.cpu_bind"),
        use_numactl=_boolean(mapping.get("numactl", False), label="defaults.numactl"),
        numa_policy=_optional_string(
            mapping.get("numa_policy"), label="defaults.numa_policy"
        ),
    )


def _parse_partition(value: object, /, *, name: str) -> PartitionLimits:
    mapping = _mapping(value, label=f"Partition {name!r}")
    _reject_unknown_keys(mapping, _PARTITION_KEYS, label=f"partition {name!r}")

    cores = mapping.get("cores_per_node")
    max_nodes = mapping.get("max_nodes")
    max_time = mapping.get("max_time")
    return PartitionLimits(
        cores_per_node=(
            None
            if cores is None
            else _positive_integer(cores, label=f"{name}.cores_per_node")
        ),
        max_nodes=(
            None
            if max_nodes is None
            else _positive_integer(max_nodes, label=f"{name}.max_nodes")
        ),
        max_minutes=(
            None
            if max_time is None
            else wall_time_minutes(
                cast(
                    str,
                    _optional_string(max_time, label=f"{name}.max_time"),
                )
            )
        ),
        requires_memory=_boolean(
            mapping.get("requires_memory", False),
            label=f"{name}.requires_memory",
        ),
        use_numactl=(
            None
            if "numactl" not in mapping
            else _boolean(mapping["numactl"], label=f"{name}.numactl")
        ),
    )


def load_cluster_profile(path: str | Path, /) -> ClusterProfile:
    source = Path(path)
    try:
        with source.open("rb") as file:
            decoded = tomllib.load(file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Cluster profile does not exist: {source}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Cluster profile is not valid TOML: {exc}") from exc

    _reject_mail_user(decoded)
    _reject_unknown_keys(decoded, _PROFILE_KEYS, label="profile")
    version = decoded.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise TypeError("Cluster profile `schema_version` must be an integer.")
    if version != PROFILE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported cluster profile schema version {version!r}; "
            f"expected {PROFILE_SCHEMA_VERSION}."
        )
    name = _optional_string(decoded.get("name"), label="Profile name")
    if name is None:
        raise ValueError("Cluster profile requires a nonempty `name`.")

    defaults = _parse_profile_defaults(decoded.get("defaults", {}))
    partitions_mapping = _mapping(
        decoded.get("partitions", {}), label="Profile partitions"
    )
    partitions: dict[str, PartitionLimits] = {}
    for partition_name, value in partitions_mapping.items():
        _validate_token(partition_name, label="Partition name")
        partitions[partition_name] = _parse_partition(value, name=partition_name)
    profile = ClusterProfile(name=name, defaults=defaults, partitions=partitions)
    _ = profile.create_config(job_name=DEFAULT_JOB_NAME)
    return profile


def resolve_profile_path(
    cli_path: str | Path | None,
    /,
    *,
    environment: Mapping[str, str] | None = None,
) -> Path | None:
    if cli_path is not None:
        return Path(cli_path)
    values = os.environ if environment is None else environment
    configured = values.get(PROFILE_ENVIRONMENT_VARIABLE)
    return None if configured is None or configured == "" else Path(configured)


def load_selected_profile(
    cli_path: str | Path | None,
    /,
    *,
    environment: Mapping[str, str] | None = None,
) -> ClusterProfile:
    path = resolve_profile_path(cli_path, environment=environment)
    return GENERIC_PROFILE if path is None else load_cluster_profile(path)
