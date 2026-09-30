import json
import os
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import cast

import attrs

from ..conversion import (
    SourceDict,
    StringEnum,
    insert_underscores,
    json_value,
    to_path,
)
from ..ensembles import RandomMatrixEnsemble
from .base_data import Data

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
        repr=False,
    )

    rng: dict[str, object] = attrs.field(
        converter=deepcopy,
        repr=False,
    )
    dtype: dict[str, str] = attrs.field(
        converter=deepcopy,
        validator=attrs.validators.instance_of(dict),
        repr=False,
    )

    execution: dict[str, object] = attrs.field(
        factory=dict,
        converter=deepcopy,
        repr=False,
    )


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class Simulation:
    _manifest: SimulationManifest = attrs.field(
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
                "initial_state": self._rmg.rng_state,
            },
            dtype={
                "real": self._rmg.real_dtype.name,
                "complex": self._rmg.complex_dtype.name,
            },
            execution={"execution_state": self._execution_state},
        )
        object.__setattr__(self, "_manifest", simulation_manifest)

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
            "name": type(self).__name__,
            "parameters": {
                field.name: json_value(cast(object, getattr(self, field.name)))
                for field in cast(
                    tuple[attrs.Attribute[object], ...], attrs.fields(type(self))
                )
                if field.init
            },
        }

    def _update_run_manifest(self, *, execution: dict[str, object]) -> None:
        if self.execution_state is not ExecutionState.RUNNING:
            raise RuntimeError("The simulation must be running.")

        self._manifest.rng["final_state"] = json_value(self._rmg.rng_state)
        self._manifest.execution["execution_state"] = self._execution_state
        self._manifest.execution.update(execution)

    def _write_manifest(self, *, path: Path) -> None:
        encoded_manifest = json.dumps(
            attrs.asdict(self._manifest),
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

    def execute(self) -> None:
        if self.execution_state is not ExecutionState.NEW:
            raise RuntimeError("A simulation instance may be executed only once.")
        else:
            object.__setattr__(self, "_execution_state", ExecutionState.RUNNING)

        try:
            self._execute()
        except BaseException:
            object.__setattr__(self, "_execution_state", ExecutionState.FAILED)
            raise

        object.__setattr__(self, "_execution_state", ExecutionState.COMPLETE)

    def save_execution(
        self,
        *,
        path: str | Path = DEFAULT_OUTPUT_ROOT,
    ) -> None:
        if self.execution_state is not ExecutionState.COMPLETE:
            raise RuntimeError("A simulation may be saved only after execution.")

        path = Path(path) / self.to_path

        completion_time = self._manifest.execution.get("execution_time")
        if not isinstance(completion_time, str):
            raise ValueError("Execution completion time is malformed.")

        destination_directory = path / Path(completion_time)
        destination_directory.mkdir(parents=True, exist_ok=True)

        self._write_manifest(path=destination_directory / Path(MANIFEST_FILE_NAME))

        for finalized_data in self:
            finalized_data.save(directory=destination_directory)
