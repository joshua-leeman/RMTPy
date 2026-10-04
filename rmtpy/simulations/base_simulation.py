import json
import os
import re
from collections.abc import Callable, Iterator
from copy import deepcopy
from pathlib import Path
from typing import cast

import attrs
import numpy as np

from ..conversion import (
    AttrsFields,
    SourceDict,
    StringEnum,
    import_rmtpy_object,
    insert_underscores,
    json_value,
    read_utc_time,
    to_key_of_registry,
    to_path,
    unwrap_json_value,
)
from ..ensembles import RandomMatrixEnsemble
from ..ensembles.base_ensemble import REGISTRY as ENSEMBLE_REGISTRY
from ..validators import is_source_dict
from .base_data import Data, graft_loaded_data, load_saved_data

DEFAULT_OUTPUT_ROOT: Path = Path("outputs")

MANIFEST_FILE_NAME: str = "manifest.json"

JOB_OUTPUT_DIRECTORY_PATTERN: re.Pattern[str] = re.compile(r"job_(?P<index>\d+)_outputs")


class ExecutionState(StringEnum):
    NEW = "new"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


@attrs.frozen(kw_only=True, eq=True, weakref_slot=False)
class SimulationManifest:
    configuration: SourceDict = attrs.field(
        converter=deepcopy,
        validator=is_source_dict,
        repr=False,
    )

    rng: dict[str, object] = attrs.field(
        converter=deepcopy,
        validator=attrs.validators.deep_mapping(
            key_validator=attrs.validators.instance_of(str),
            mapping_validator=attrs.validators.instance_of(dict),
        ),
        repr=False,
    )

    dtype: dict[str, str] = attrs.field(
        converter=deepcopy,
        validator=attrs.validators.deep_mapping(
            key_validator=attrs.validators.instance_of(str),
            value_validator=attrs.validators.instance_of(str),
            mapping_validator=attrs.validators.instance_of(dict),
        ),
        repr=False,
    )

    execution: dict[str, object] = attrs.field(
        factory=cast(Callable[..., dict[str, object]], dict),
        converter=deepcopy,
        validator=attrs.validators.deep_mapping(
            key_validator=attrs.validators.instance_of(str),
            mapping_validator=attrs.validators.instance_of(dict),
        ),
        repr=False,
    )

    @classmethod
    def create_manifest_from_path(cls, path: str | Path) -> SimulationManifest:
        manifest_text = Path(path).read_text(encoding="utf-8")
        manifest = cast(dict[str, object], json.loads(manifest_text))

        return SimulationManifest(
            configuration=cast(SourceDict, manifest.get("configuration")),
            rng=cast(dict[str, object], manifest.get("rng")),
            dtype=cast(dict[str, str], manifest.get("dtype")),
            execution=cast(dict[str, object], manifest.get("execution")),
        )


def _structure_argument(value: object) -> object:
    unwrapped_value = unwrap_json_value(value)
    if not isinstance(unwrapped_value, dict):
        return unwrapped_value

    source = cast(SourceDict, unwrapped_value)
    type_name = source.get("type")
    if (
        set(source) == {"type", "parameters"}
        and isinstance(type_name, str)
        and to_key_of_registry(type_name) in ENSEMBLE_REGISTRY
    ):
        return RandomMatrixEnsemble.create(source)

    return cast(dict[str, object], unwrapped_value)


def _simulation_class_from_manifest(
    manifest: SimulationManifest,
) -> type[Simulation]:
    module_name = cast(str, manifest.configuration["module"])
    class_name = cast(str, manifest.configuration["type"])
    simulation_cls = import_rmtpy_object(class_name, module_name=module_name)
    if not isinstance(simulation_cls, type):
        raise ValueError("Imported object is not a type.")
    if not issubclass(simulation_cls, Simulation):
        raise ValueError("Imported class is not a Simulation.")

    return simulation_cls


def _replace_nested_seeds(value: object, replacement: object) -> object:
    if isinstance(value, dict):
        return {
            key: (
                deepcopy(replacement)
                if key == "seed"
                else _replace_nested_seeds(item, replacement)
            )
            for key, item in cast(dict[str, object], value).items()
        }
    if isinstance(value, list):
        return [
            _replace_nested_seeds(item, replacement) for item in cast(list[object], value)
        ]

    return deepcopy(value)


def _aggregation_configuration(manifest: SimulationManifest) -> SourceDict:
    configuration = deepcopy(manifest.configuration)
    parameters = cast(dict[str, object], configuration["parameters"])
    normalized_parameters = cast(
        dict[str, object],
        _replace_nested_seeds(parameters, None),
    )
    _ = normalized_parameters.pop("realizs", None)
    _ = normalized_parameters.pop("_execution_state", None)
    return {
        "module": cast(str, configuration["module"]),
        "type": cast(str, configuration["type"]),
        "parameters": normalized_parameters,
    }


def _create_simulation_from_manifest(
    manifest: SimulationManifest,
    *,
    realizs: int | None = None,
    reset_seed: bool = False,
) -> Simulation:
    simulation_cls = _simulation_class_from_manifest(manifest)
    parameters = deepcopy(cast(dict[str, object], manifest.configuration["parameters"]))
    if realizs is not None:
        parameters["realizs"] = realizs
    if reset_seed:
        parameters = cast(dict[str, object], _replace_nested_seeds(parameters, None))

    arguments: dict[str, object] = {}
    for field in cast(AttrsFields, attrs.fields(simulation_cls)):
        if not field.init or field.default is not attrs.NOTHING:
            continue
        if field.name not in parameters:
            raise ValueError(f"Manifest parameters are missing `{field.name}`.")

        arguments[field.name] = _structure_argument(parameters[field.name])

    simulation_factory = cast(Callable[..., Simulation], simulation_cls)
    return simulation_factory(**arguments)


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class Simulation:
    manifest: SimulationManifest = attrs.field(
        init=False,
    )

    _execution_state: ExecutionState = attrs.field(
        default=ExecutionState.NEW,
    )

    def __attrs_post_init__(self) -> None:
        simulation_manifest = SimulationManifest(
            configuration=self._build_configuration(),
            rng={
                "policy": "numpy.random.default_rng",
                "bit_generator": type(self._rmg.rng.bit_generator).__name__,
                "seed": json_value(self._rmg.seed),
                "state_policy": "capture_initial_and_final",
                "initial_state": json_value(self._rmg.rng_state),
            },
            dtype={
                "real": self._rmg.real_dtype.name,
                "complex": self._rmg.complex_dtype.name,
            },
            execution={"execution_state": self._execution_state},
        )
        object.__setattr__(self, "manifest", simulation_manifest)

    def __iter__(self) -> Iterator[Data]:
        raise NotImplementedError(
            f"{type(self).__name__} has not implemented iterator method."
        )

    @property
    def token_name(self) -> str:
        return insert_underscores(type(self).__name__).lower()

    @property
    def _rmg(self) -> RandomMatrixEnsemble:
        raise NotImplementedError(
            f"{type(self).__name__} has not implemented property `_rmg`."
        )

    @property
    def _root_for_outputs(self) -> Path:
        return Path(self.token_name)

    @property
    def to_path(self) -> Path:
        return to_path(self, root=self._root_for_outputs)

    @property
    def execution_state(self) -> ExecutionState:
        return self._execution_state

    def _build_configuration(self) -> SourceDict:
        return {
            "module": type(self).__module__,
            "type": type(self).__name__,
            "parameters": {
                field.name: json_value(cast(object, getattr(self, field.name)))
                for field in cast(AttrsFields, attrs.fields(type(self)))
                if field.init and field.repr
            },
        }

    def _write_manifest(self, *, path: Path) -> None:
        encoded_manifest = json.dumps(
            attrs.asdict(self.manifest),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=False,
        )
        with path.open("x", encoding="utf-8") as file:
            _ = file.write(encoded_manifest)
            _ = file.write("\n")
            file.flush()
            os.fsync(file.fileno())

    def _execute(self) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} has not implemented the execution program."
        )

    def _finalize(self) -> None:
        for data in self:
            data.compute_statistics()

    def _restore_execution(self) -> None:
        pass

    def execute(self) -> None:
        if self.execution_state is not ExecutionState.NEW:
            raise RuntimeError("A simulation instance may be executed only once.")

        object.__setattr__(self, "_execution_state", ExecutionState.RUNNING)
        try:
            self._execute()
        except BaseException:
            object.__setattr__(self, "_execution_state", ExecutionState.FAILED)
            raise

        object.__setattr__(self, "_execution_state", ExecutionState.COMPLETE)

        self.manifest.rng["final_state"] = json_value(self._rmg.rng_state)
        self.manifest.execution.update(
            {key: json_value(value) for key, value in self.manifest.execution.items()}
        )
        self.manifest.execution["execution_state"] = self._execution_state
        self.manifest.execution["execution_time"] = read_utc_time()

    def save(self, directory: str | Path = DEFAULT_OUTPUT_ROOT, /) -> Path:
        if self.execution_state is not ExecutionState.COMPLETE:
            raise RuntimeError("A simulation may be saved only after execution.")

        completion_time = self.manifest.execution.get("execution_time")
        if not isinstance(completion_time, str) or completion_time == "":
            raise ValueError("Execution completion time is malformed.")

        destination_directory = Path(directory) / self.to_path / completion_time
        manifest_path = destination_directory / MANIFEST_FILE_NAME
        if manifest_path.exists():
            raise FileExistsError(f"`{manifest_path}` already exists.")

        destination_directory.mkdir(parents=True, exist_ok=True)
        for finalized_data in self:
            finalized_data.save(directory=destination_directory)

        self._write_manifest(path=manifest_path)
        return destination_directory

    def plot(self, _directory: str | Path, /) -> None:
        raise NotImplementedError(f"{type(self).__name__} has not implemented plotting.")

    @classmethod
    def aggregate[SimulationType: Simulation](
        cls: type[SimulationType],
        superfolder: str | Path,
        /,
    ) -> SimulationType:
        superfolder = Path(superfolder)
        if not superfolder.is_dir():
            raise ValueError(f"Aggregation superfolder `{superfolder}` is malformed.")

        indexed_job_directories: list[tuple[int, Path]] = []
        for child in superfolder.iterdir():
            match = JOB_OUTPUT_DIRECTORY_PATTERN.fullmatch(child.name)
            if child.is_dir() and match is not None:
                indexed_job_directories.append((int(match.group("index")), child))

        if not indexed_job_directories:
            raise ValueError(
                f"Aggregation superfolder `{superfolder}` has no job output directories."
            )

        indexed_job_directories.sort(key=lambda item: (item[0], item[1].name))
        job_indices = [index for index, _ in indexed_job_directories]
        if len(job_indices) != len(set(job_indices)):
            raise ValueError("Job output directory indices must be unique.")

        source_records: list[tuple[int, Path, SimulationManifest, type[Simulation]]] = []
        for job_index, job_directory in indexed_job_directories:
            candidates: list[tuple[Path, SimulationManifest, type[Simulation]]] = []
            for manifest_path in sorted(job_directory.rglob(MANIFEST_FILE_NAME)):
                manifest = SimulationManifest.create_manifest_from_path(manifest_path)
                simulation_cls = _simulation_class_from_manifest(manifest)
                if cls is Simulation or cls in simulation_cls.__mro__:
                    candidates.append((manifest_path.parent, manifest, simulation_cls))

            if len(candidates) != 1:
                requested_name = (
                    "an unambiguous simulation"
                    if cls is Simulation
                    else f"one {cls.__name__}"
                )
                raise ValueError(
                    f"Job directory `{job_directory}` must contain exactly "
                    + f"{requested_name}; found {len(candidates)}."
                )

            source_directory, manifest, simulation_cls = candidates[0]
            source_records.append((job_index, source_directory, manifest, simulation_cls))

        first_manifest = source_records[0][2]
        first_simulation_cls = source_records[0][3]
        normalized_configuration = _aggregation_configuration(first_manifest)
        expected_dtype = first_manifest.dtype
        source_realizs: list[int] = []
        for _, source_directory, manifest, simulation_cls in source_records:
            execution_state = manifest.execution.get("execution_state")
            if execution_state != ExecutionState.COMPLETE:
                raise ValueError(
                    f"Simulation `{source_directory}` is not complete and cannot "
                    + "be aggregated."
                )
            if simulation_cls is not first_simulation_cls:
                raise ValueError("Job outputs contain different simulation classes.")
            if _aggregation_configuration(manifest) != normalized_configuration:
                raise ValueError(
                    f"Simulation `{source_directory}` has an incompatible "
                    + "scientific configuration."
                )
            if manifest.dtype != expected_dtype:
                raise ValueError(
                    f"Simulation `{source_directory}` has incompatible dtypes."
                )

            completion_time = manifest.execution.get("execution_time")
            if not isinstance(completion_time, str) or completion_time == "":
                raise ValueError(
                    f"Simulation `{source_directory}` has a malformed completion time."
                )

            parameters = cast(dict[str, object], manifest.configuration["parameters"])
            realizs = parameters.get("realizs")
            if not isinstance(realizs, int) or isinstance(realizs, bool) or realizs < 1:
                raise ValueError(
                    f"Simulation `{source_directory}` has an invalid realization count."
                )
            source_realizs.append(realizs)

        total_realizs = sum(source_realizs)
        aggregate = _create_simulation_from_manifest(
            first_manifest,
            realizs=total_realizs,
            reset_seed=True,
        )

        aggregate_data = {data.aggregation_key: data for data in aggregate}
        if len(aggregate_data) != len(tuple(aggregate)):
            raise ValueError("Aggregate simulation contains duplicate data names.")

        source_simulations: list[Simulation] = []
        for (_, source_directory, _, _), expected_realizs in zip(
            source_records,
            source_realizs,
            strict=True,
        ):
            source = Simulation.load(source_directory)
            source_simulations.append(source)
            source_data = {data.aggregation_key: data for data in source}
            if len(source_data) != len(tuple(source)):
                raise ValueError(
                    f"Simulation `{source_directory}` contains duplicate data names."
                )
            if set(source_data) != set(aggregate_data):
                raise ValueError(
                    f"Simulation `{source_directory}` has an incompatible data layout."
                )
            for data in source_data.values():
                if data.realizs != expected_realizs:
                    raise ValueError(
                        f"Saved data `{data._file_name}` has {data.realizs} "
                        + f"realizations; expected {expected_realizs}."
                    )

        for source in source_simulations:
            source_data = {data.aggregation_key: data for data in source}
            for file_name, data in aggregate_data.items():
                data.add_contribution(source_data[file_name])

        for data in aggregate_data.values():
            if data.realizs != total_realizs:
                raise ValueError(
                    f"Aggregated data `{data._file_name}` has {data.realizs} "
                    + f"realizations; expected {total_realizs}."
                )

        aggregate._finalize()

        calibrations: list[dict[str, object]] = []
        for _, _, manifest, _ in source_records:
            calibration = unwrap_json_value(manifest.execution.get("calibration", {}))
            if not isinstance(calibration, dict):
                raise ValueError("Saved simulation calibration is malformed.")
            calibrations.append(cast(dict[str, object], calibration))

        nonempty_calibrations = [
            calibration for calibration in calibrations if calibration
        ]
        aggregate_calibration: dict[str, object] = {}
        if nonempty_calibrations:
            if len(nonempty_calibrations) != len(calibrations):
                raise ValueError("Saved simulations have inconsistent calibrations.")

            densities = {calibration.get("density") for calibration in calibrations}
            if len(densities) != 1 or not all(
                isinstance(density, str) for density in densities
            ):
                raise ValueError("Saved simulations have incompatible calibrations.")

            coefficient_arrays = [
                np.asarray(calibration.get("average_coefficients"), dtype=np.float64)
                for calibration in calibrations
            ]
            first_shape = coefficient_arrays[0].shape
            if not first_shape or any(
                coefficients.shape != first_shape or not np.all(np.isfinite(coefficients))
                for coefficients in coefficient_arrays
            ):
                raise ValueError("Saved simulations have malformed calibrations.")

            pooled_coefficients = np.average(
                np.stack(coefficient_arrays),
                axis=0,
                weights=np.asarray(source_realizs, dtype=np.float64),
            )
            aggregate_calibration = {
                "density": densities.pop(),
                "average_coefficients": pooled_coefficients,
                "timing": "aggregated",
            }

        completion_time = read_utc_time()
        source_seeds = [
            deepcopy(manifest.rng.get("seed")) for _, _, manifest, _ in source_records
        ]
        aggregate.manifest.rng.update(
            {
                "policy": "aggregate",
                "seed": {
                    "type": "aggregate",
                    "source_seeds": source_seeds,
                },
                "final_state": json_value(aggregate._rmg.rng_state),
            }
        )
        aggregate.manifest.execution.clear()
        aggregate.manifest.execution.update(
            {
                "execution_state": ExecutionState.COMPLETE,
                "execution_time": completion_time,
                "calibration": json_value(aggregate_calibration),
                "aggregation": {
                    "source_directories": [
                        str(source_directory.relative_to(superfolder))
                        for _, source_directory, _, _ in source_records
                    ],
                    "job_indices": [job_index for job_index, _, _, _ in source_records],
                    "source_realizs": source_realizs,
                    "source_calibrations": json_value(calibrations),
                    "unfolding_policy": "pooled_source_calibrations",
                    "source_completion_times": [
                        manifest.execution.get("execution_time")
                        for _, _, manifest, _ in source_records
                    ],
                },
            }
        )
        object.__setattr__(aggregate, "_execution_state", ExecutionState.COMPLETE)
        aggregate._restore_execution()
        return cast(SimulationType, aggregate)

    @classmethod
    def load[SimulationType: Simulation](
        cls: type[SimulationType],
        directory: str | Path,
        /,
    ) -> SimulationType:
        directory = Path(directory)
        manifest_path = directory / MANIFEST_FILE_NAME
        if not directory.is_dir() or not manifest_path.is_file():
            raise ValueError(f"Simulation directory `{directory}` is malformed.")

        manifest = SimulationManifest.create_manifest_from_path(manifest_path)

        simulation = _create_simulation_from_manifest(manifest)
        if cls is not Simulation and not isinstance(simulation, cls):
            raise TypeError(f"Saved simulation is not a {cls.__name__}.")

        loaded_data = load_saved_data(directory)
        _ = graft_loaded_data(simulation, loaded_data)
        if len(loaded_data) > 0:
            file_name = sorted(loaded_data)[0]
            raise ValueError(f"Saved data `{file_name}` is not part of the simulation.")

        object.__setattr__(simulation, "manifest", manifest)

        execution_state = manifest.execution.get("execution_state")
        if not isinstance(execution_state, str):
            raise ValueError("Saved execution state is not of type str.")
        if not ExecutionState.has_value(execution_state):
            raise ValueError("Saved execution state is not valid.")

        object.__setattr__(
            simulation,
            "_execution_state",
            ExecutionState(execution_state),
        )

        rng_final_state = unwrap_json_value(manifest.rng.get("final_state"))
        if not isinstance(rng_final_state, dict):
            raise ValueError("RNG final state is malformed.")

        simulation._rmg.rng.bit_generator.state = rng_final_state
        simulation._restore_execution()

        return cast(SimulationType, simulation)
