from collections.abc import Container
from copy import deepcopy
from pathlib import Path
from typing import Protocol, cast

import attrs

from ..conversion import (
    SourceDict,
    StringEnum,
    get_attrs_fields,
    insert_underscores,
    to_json_compatible,
    to_path,
)
from ..ensembles.base_ensemble import RandomMatrixEnsemble, SeedLike


class _EnsembleRngOwner(Protocol):
    ensemble: RandomMatrixEnsemble


class ExecutionState(StringEnum):
    NEW = "new"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


@attrs.frozen(kw_only=True, eq=True, weakref_slot=False)
class RunActivation:
    configuration: SourceDict = attrs.field(converter=deepcopy)

    rng_seed: SeedLike = attrs.field(converter=deepcopy)
    rng_state: dict[str, object] = attrs.field(converter=deepcopy, repr=False)


@attrs.frozen(kw_only=True, eq=True, weakref_slot=False)
class RunContext:
    simulation_type: str
    result_type: str

    configuration: SourceDict = attrs.field(converter=deepcopy)

    requested_outputs: dict[str, object] = attrs.field(converter=deepcopy)

    rng: dict[str, object] = attrs.field(converter=deepcopy, repr=False)
    dtype: dict[str, str] = attrs.field(converter=deepcopy)

    execution: dict[str, object] = attrs.field(factory=dict, converter=deepcopy)


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class Result:
    context: RunContext


def _resolve_rng_from_ensemble(
    rng_owner: RandomMatrixEnsemble | _EnsembleRngOwner,
    /,
) -> RandomMatrixEnsemble:
    if isinstance(rng_owner, RandomMatrixEnsemble):
        return rng_owner

    return cast(_EnsembleRngOwner, rng_owner).ensemble


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class Simulation:
    _execution_state: ExecutionState = attrs.field(
        default=ExecutionState.NEW,
        init=False,
        repr=False,
    )
    _result: Result = attrs.field(
        default=None,
        init=False,
        repr=False,
    )

    @property
    def execution_state(self) -> ExecutionState:
        return self._execution_state

    @property
    def result(self) -> Result:
        if self._execution_state is not ExecutionState.COMPLETE:
            raise RuntimeError("The simulation has not completely executed yet.")

        return self._result

    @property
    def root_for_outputs(self) -> Path:
        return Path(insert_underscores(type(self).__name__).lower())

    @property
    def to_path(self) -> Path:
        return to_path(self, root=self.root_for_outputs)

    def execute(self) -> Result:
        if self._execution_state is not ExecutionState.NEW:
            raise RuntimeError("A simulation instance may be executed only once.")
        else:
            object.__setattr__(self, "_execution_state", ExecutionState.RUNNING)

        try:
            object.__setattr__(self, "_result", self._execute())
        except BaseException:
            object.__setattr__(self, "_execution_state", ExecutionState.FAILED)
            raise

        object.__setattr__(self, "_execution_state", ExecutionState.COMPLETE)
        return self._result

    def _record_configuration(self, *, fields_to_omit: Container[str] = ()) -> SourceDict:
        fields = get_attrs_fields(type(self))
        return {
            "type": type(self).__name__,
            "parameters": {
                field.name: to_json_compatible(cast(object, getattr(self, field.name)))
                for field in fields
                if field.init and field.name not in fields_to_omit
            },
        }

    def _capture_run_activation(
        self,
        *,
        rng_owner: RandomMatrixEnsemble | _EnsembleRngOwner,
        requested_outputs: Container[str] = (),
    ) -> RunActivation:
        if self._execution_state is not ExecutionState.RUNNING:
            raise RuntimeError("The simulation must be running.")

        ensemble = _resolve_rng_from_ensemble(rng_owner)
        return RunActivation(
            configuration=self._record_configuration(fields_to_omit=requested_outputs),
            rng_seed=cast(SeedLike, to_json_compatible(ensemble.seed)),
            rng_state=cast(dict[str, object], to_json_compatible(ensemble.rng_state)),
        )

    def _establish_run_context(
        self,
        *,
        result_type: str,
        rng_owner: RandomMatrixEnsemble | _EnsembleRngOwner,
        run_activation: RunActivation,
        requested_outputs: dict[str, object],
        execution: dict[str, object] | None = None,
    ) -> RunContext:
        if self._execution_state is not ExecutionState.RUNNING:
            raise RuntimeError("The simulation must be running.")

        ensemble = _resolve_rng_from_ensemble(rng_owner)

        return RunContext(
            simulation_type=insert_underscores(type(self).__name__).lower(),
            result_type=result_type,
            configuration=run_activation.configuration,
            requested_outputs=requested_outputs,
            rng={
                "policy": "numpy.random.default_rng",
                "bit_generator": type(ensemble.rng.bit_generator).__name__,
                "seed": run_activation.rng_seed,
                "state_policy": "capture_initial_and_final",
                "initial_state": run_activation.rng_state,
                "final_state": to_json_compatible(ensemble.rng_state),
            },
            dtype={
                "configured": ensemble.dtype.name,
                "real": ensemble.real_dtype.name,
                "complex": ensemble.complex_dtype.name,
            },
            execution=({} if execution is None else execution),
        )

    def _execute(self) -> Result:
        raise NotImplementedError(
            f"{type(self).__name__} has not implemented the execution program."
        )
