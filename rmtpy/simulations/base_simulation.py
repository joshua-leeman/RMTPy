import json
import os
from collections.abc import Callable, Iterator
from copy import deepcopy
from pathlib import Path
from typing import cast

import attrs

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


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class Simulation:
    manifest: SimulationManifest = attrs.field(
        init=False,
        repr=False,
    )

    _execution_state: ExecutionState = attrs.field(
        default=ExecutionState.NEW,
        repr=False,
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
    def _token_name(self) -> str:
        return insert_underscores(type(self).__name__).lower()

    @property
    def _rmg(self) -> RandomMatrixEnsemble:
        raise NotImplementedError(
            f"{type(self).__name__} has not implemented property `_rmg`."
        )

    @property
    def _root_for_outputs(self) -> Path:
        return Path(self._token_name)

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
                if field.init
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

    @classmethod
    def load(cls, directory: str | Path, /) -> Simulation:
        directory = Path(directory)
        manifest_path = directory / MANIFEST_FILE_NAME
        if not directory.is_dir() or not manifest_path.is_file():
            raise ValueError(f"Simulation directory `{directory}` is malformed.")

        manifest = SimulationManifest.create_manifest_from_path(manifest_path)

        module_name = cast(str, manifest.configuration["module"])
        class_name = cast(str, manifest.configuration["type"])
        simulation_cls = import_rmtpy_object(class_name, module_name=module_name)
        if not isinstance(simulation_cls, type):
            raise ValueError("Imported object is not a type.")
        if not issubclass(simulation_cls, Simulation):
            raise ValueError("Imported class is not a Simulation.")

        parameters = cast(dict[str, object], manifest.configuration["parameters"])
        arguments: dict[str, object] = {}
        for field in cast(AttrsFields, attrs.fields(simulation_cls)):
            if not field.init or field.default is not attrs.NOTHING:
                continue
            if field.name not in parameters:
                raise ValueError(f"Manifest parameters are missing `{field.name}`.")

            arguments[field.name] = _structure_argument(parameters[field.name])

        simulation_factory = cast(Callable[..., Simulation], simulation_cls)
        simulation = simulation_factory(**arguments)

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

        return simulation
