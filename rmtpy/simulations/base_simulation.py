from copy import deepcopy
from typing import ClassVar, cast

import attrs

from ..conversion import SourceDict, StringEnum, insert_underscores, to_json_compatible
from ..ensembles.base_ensemble import RandomMatrixEnsemble, SeedLike


class SimulationExecutionState(StringEnum):
    NEW = "new"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


@attrs.frozen(kw_only=True, eq=True, weakref_slot=False)
class RunActivation:
    simulation_config: SourceDict = attrs.field(converter=deepcopy)

    rng_seed: SeedLike = attrs.field(converter=deepcopy)
    rng_state: dict[str, object] = attrs.field(converter=deepcopy, repr=False)


@attrs.frozen(kw_only=True, eq=True, weakref_slot=False)
class RunContext:
    simulation_type: str
    result_type: str

    simulation_config: SourceDict = attrs.field(converter=deepcopy)

    output_request: dict[str, object] = attrs.field(converter=deepcopy)

    rng: dict[str, object] = attrs.field(converter=deepcopy, repr=False)
    dtype: dict[str, str] = attrs.field(converter=deepcopy)

    execution: dict[str, object] = attrs.field(factory=dict, converter=deepcopy)


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class Result:
    context: RunContext


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class Simulation:
    ExecutionState: ClassVar[type[SimulationExecutionState]] = SimulationExecutionState

    _execution_state: SimulationExecutionState = attrs.field(
        default=SimulationExecutionState.NEW,
        init=False,
        repr=False,
    )
    _result: Result = attrs.field(
        default=None,
        init=False,
        repr=False,
    )

    @property
    def execution_state(self) -> SimulationExecutionState:
        return self._execution_state

    @property
    def result(self) -> Result:
        if self._execution_state is not SimulationExecutionState.COMPLETE:
            raise RuntimeError("The simulation has not completed successfully.")

        return self._result

    def _assemble_configuration(
        self,
        *,
        output_fields: tuple[str, ...] = (),
    ) -> SourceDict:
        fields = cast(tuple[attrs.Attribute[object], ...], attrs.fields(type(self)))
        return {
            "type": type(self).__name__,
            "parameters": {
                field.name: to_json_compatible(cast(object, getattr(self, field.name)))
                for field in fields
                if field.init and field.name not in output_fields
            },
        }

    def _capture_run_activation(
        self,
        *,
        rng_owner: object,
        output_fields: tuple[str, ...] = (),
    ) -> RunActivation:
        if self._execution_state is not SimulationExecutionState.NEW:
            raise RuntimeError("The simulation must be newly created.")

        ensemble = cast(RandomMatrixEnsemble, getattr(rng_owner, "ensemble", rng_owner))
        return RunActivation(
            simulation_config=self._assemble_configuration(output_fields=output_fields),
            rng_seed=cast(SeedLike, to_json_compatible(ensemble.seed)),
            rng_state=cast(dict[str, object], to_json_compatible(ensemble.rng_state)),
        )

    def _build_run_context(
        self,
        *,
        result_type: str,
        rng_owner: object,
        run_start: RunActivation,
        output_request: dict[str, object],
        execution: dict[str, object] | None = None,
    ) -> RunContext:
        if self._execution_state is not SimulationExecutionState.COMPLETE:
            raise RuntimeError("The simulation must be complete.")

        ensemble = cast(RandomMatrixEnsemble, getattr(rng_owner, "ensemble", rng_owner))
        bit_generator = ensemble.rng.bit_generator
        output_request = cast(dict[str, object], to_json_compatible(output_request))
        return RunContext(
            simulation_type=insert_underscores(type(self).__name__).lower(),
            result_type=result_type,
            simulation_config=run_start.simulation_config,
            output_request=output_request,
            rng={
                "policy": "numpy.random.default_rng",
                "bit_generator": type(bit_generator).__name__,
                "seed": run_start.rng_seed,
                "state_policy": "capture_initial_and_final",
                "initial_state": run_start.rng_state,
                "final_state": to_json_compatible(ensemble.rng_state),
            },
            dtype={
                "configured": ensemble.dtype.name,
                "real": ensemble.real_dtype.name,
                "complex": ensemble.complex_dtype.name,
            },
            execution=({} if execution is None else output_request),
        )

    def _execute(self) -> Result:
        raise NotImplementedError(
            f"{type(self).__name__} has not implemented the execution contract."
        )

    def execute(self) -> Result:
        if self._execution_state is not SimulationExecutionState.NEW:
            raise RuntimeError("A simulation instance may be executed only once.")

        object.__setattr__(self, "_execution_state", SimulationExecutionState.RUNNING)
        try:
            object.__setattr__(self, "_result", self._execute())
        except BaseException:
            object.__setattr__(self, "_execution_state", SimulationExecutionState.FAILED)
            raise

        object.__setattr__(self, "_execution_state", SimulationExecutionState.COMPLETE)
        return self._result
