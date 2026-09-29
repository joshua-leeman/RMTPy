from copy import deepcopy
from pathlib import Path
from typing import cast

import attrs

from ..conversion import (
    SourceDict,
    StringEnum,
    get_attrs_fields,
    insert_underscores,
    to_json_compatible,
    to_path,
)
from ..ensembles.base_ensemble import RandomMatrixEnsemble


class ExecutionState(StringEnum):
    NEW = "new"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


@attrs.frozen(kw_only=True, eq=True, weakref_slot=False)
class SimulationContext:
    simulation: str = attrs.field(
        validator=attrs.validators.instance_of(str),
    )
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
    _context: SimulationContext = attrs.field(
        init=False,
        repr=False,
    )
    _execution_state: ExecutionState = attrs.field(
        default=ExecutionState.NEW,
        repr=False,
    )

    def __attrs_post_init__(self) -> None:
        simulation_context = SimulationContext(
            simulation=self._token_name,
            configuration=self._build_configuration(),
            rng={
                "policy": "numpy.random.default_rng",
                "bit_generator": type(self._rmg.rng.bit_generator).__name__,
                "seed": to_json_compatible(self._rmg.seed),
                "state_policy": "capture_initial_and_final",
                "initial_state": self._rng_initial_state,
            },
            dtype={
                "real": self._rmg.real_dtype.name,
                "complex": self._rmg.complex_dtype.name,
            },
        )
        object.__setattr__(self, "_context", simulation_context)

    @property
    def _token_name(self) -> str:
        return insert_underscores(type(self).__name__).lower()

    @property
    def _root_for_outputs(self) -> Path:
        return Path(self._token_name)

    @property
    def _rmg(self) -> RandomMatrixEnsemble:
        raise NotImplementedError(
            f"{type(self).__name__} has not implemented property `_rmg`."
        )

    @property
    def _rng_initial_state(self) -> dict[str, object]:
        raise NotImplementedError(
            f"{type(self).__name__} has not implemented property `_rng_initial_states`."
        )

    @property
    def execution_state(self) -> ExecutionState:
        return self._execution_state

    @property
    def to_path(self) -> Path:
        return to_path(self, root=self._root_for_outputs)

    def _build_configuration(self) -> SourceDict:
        return {
            "name": type(self).__name__,
            "parameters": {
                field.name: to_json_compatible(cast(object, getattr(self, field.name)))
                for field in get_attrs_fields(type(self))
                if field.init
            },
        }

    def _store_run_context(self, *, execution: dict[str, object]) -> None:
        if self.execution_state is not ExecutionState.RUNNING:
            raise RuntimeError("The simulation must be running.")

        self._context.rng["final_state"] = to_json_compatible(self._rmg.rng_state)
        self._context.execution.update(execution)

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
            object.__setattr__(self, "_result", self._execute())
        except BaseException:
            object.__setattr__(self, "_execution_state", ExecutionState.FAILED)
            raise

        object.__setattr__(self, "_execution_state", ExecutionState.COMPLETE)
