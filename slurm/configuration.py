import ast
import json
import re
import secrets
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import cast

import attrs
import numpy as np

from rmtpy.compounds import (
    CompoundEnsemble,
    PoissonCompoundEnsemble,
    SYKCompoundEnsemble,
)
from rmtpy.conversion import SourceDict, json_value, source_dict, unwrap_json_value
from rmtpy.ensembles import (
    GOE,
    GSE,
    GUE,
    SYK,
    BdGC,
    BdGD,
    ManyBodyEnsemble,
    Poisson,
)
from rmtpy.simulations import (
    PartialWidthsStatisticsSimulation,
    ResonanceStatisticsSimulation,
    Simulation,
    SpectralStatisticsSimulation,
    TimeDelayStatisticsSimulation,
    TransmissionCoefficientsSimulation,
)

type MasterSeed = int | list[int]


def normalize_name(value: str, /) -> str:
    return re.sub(r"[\s_-]+", "", value).lower()


ENSEMBLE_TYPES: dict[str, type[ManyBodyEnsemble]] = {
    "goe": GOE,
    "gaussianorthogonalensemble": GOE,
    "gue": GUE,
    "gaussianunitaryensemble": GUE,
    "gse": GSE,
    "gaussiansymplecticensemble": GSE,
    "bdgc": BdGC,
    "bogoliubovdegennescensemble": BdGC,
    "bdgd": BdGD,
    "bogoliubovdegennesdensemble": BdGD,
    "poisson": Poisson,
    "poissonensemble": Poisson,
    "syk": SYK,
    "sachdevyekitaev": SYK,
    "sachdevyekitaevensemble": SYK,
}

COMPOUND_TYPES: dict[str, type[CompoundEnsemble]] = {
    "compound": CompoundEnsemble,
    "compoundensemble": CompoundEnsemble,
    "poissoncompound": PoissonCompoundEnsemble,
    "poissoncompoundensemble": PoissonCompoundEnsemble,
    "sykcompound": SYKCompoundEnsemble,
    "sykcompoundensemble": SYKCompoundEnsemble,
}

SIMULATION_TYPES: dict[str, type[Simulation]] = {
    "spectralstatistics": SpectralStatisticsSimulation,
    "resonancestatistics": ResonanceStatisticsSimulation,
    "partialwidthsstatistics": PartialWidthsStatisticsSimulation,
    "timedelaystatistics": TimeDelayStatisticsSimulation,
    "transmissioncoefficients": TransmissionCoefficientsSimulation,
}

SIMULATION_NAMES_BY_CLASS: dict[type[Simulation], str] = {
    simulation_cls: name for name, simulation_cls in SIMULATION_TYPES.items()
}

SIMULATION_ARGUMENTS: dict[str, tuple[set[str], set[str]]] = {
    "spectralstatistics": (set(), set()),
    "resonancestatistics": (set(), set()),
    "partialwidthsstatistics": ({"width_indices"}, {"width_indices"}),
    "timedelaystatistics": (
        {"energies", "spectral_statistics_directory"},
        {"energies"},
    ),
    "transmissioncoefficients": ({"channel_indices"}, {"channel_indices"}),
}


def parse_mapping(value: str | None, /, *, label: str) -> dict[str, object]:
    if value is None:
        return {}

    try:
        decoded = cast(object, json.loads(value))
    except json.JSONDecodeError:
        try:
            decoded = cast(object, ast.literal_eval(value))
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"{label} must be a JSON or Python-literal mapping."
            ) from exc

    if not isinstance(decoded, dict) or not all(
        isinstance(key, str) for key in cast(dict[object, object], decoded)
    ):
        raise TypeError(f"{label} must be a mapping with string keys.")

    return cast(dict[str, object], decoded)


def _pop_name(mapping: dict[str, object], /, *, label: str) -> str | None:
    names = [key for key in ("name", "type") if key in mapping]
    if len(names) > 1:
        raise ValueError(f"{label} may contain only one of `name` or `type`.")
    if not names:
        return None

    value = mapping.pop(names[0])
    if not isinstance(value, str) or value.strip() == "":
        raise TypeError(f"{label} name must be a nonempty string.")
    return value


def _rename_aliases(
    mapping: dict[str, object],
    aliases: Mapping[str, str],
    /,
) -> None:
    for alias, canonical_name in aliases.items():
        if alias not in mapping:
            continue
        if canonical_name in mapping:
            raise ValueError(
                f"Use only one of `{alias}` and `{canonical_name}` in a mapping."
            )
        mapping[canonical_name] = mapping.pop(alias)


def _normalize_master_seed(seed: object) -> MasterSeed:
    if seed is None:
        seed = secrets.randbits(128)
    if isinstance(seed, bool):
        raise TypeError("Distributed simulation seed cannot be boolean.")
    if isinstance(seed, int):
        normalized: MasterSeed = seed
    elif isinstance(seed, Sequence) and not isinstance(seed, str | bytes):
        normalized_items = list(seed)
        if not normalized_items or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in normalized_items
        ):
            raise TypeError("Distributed simulation seed sequence must contain integers.")
        normalized = cast(list[int], normalized_items)
    else:
        raise TypeError("Distributed simulation seed must be an integer or integer list.")

    _ = np.random.SeedSequence(normalized)
    return normalized


def _build_ensemble(
    mapping: Mapping[str, object],
) -> tuple[ManyBodyEnsemble, MasterSeed]:
    arguments = dict(mapping)
    ensemble_name = _pop_name(arguments, label="Ensemble")
    if ensemble_name is None:
        raise ValueError("Ensemble input requires a `name`.")

    _rename_aliases(
        arguments,
        {
            "N": "num_majoranas",
            "J": "interaction_strength",
            "max_degree": "max_spectral_polynomial_degree",
            "parity": "is_even_parity",
        },
    )
    master_seed = _normalize_master_seed(arguments.get("seed"))
    arguments["seed"] = master_seed

    try:
        ensemble_cls = ENSEMBLE_TYPES[normalize_name(ensemble_name)]
    except KeyError as exc:
        choices = ", ".join(("GOE", "GUE", "GSE", "BdGC", "BdGD", "Poisson", "SYK"))
        raise ValueError(
            f"Unknown ensemble {ensemble_name!r}; choose one of {choices}."
        ) from exc

    try:
        ensemble_factory = cast(Callable[..., ManyBodyEnsemble], ensemble_cls)
        ensemble = ensemble_factory(**arguments)
    except TypeError as exc:
        raise TypeError(f"Invalid parameters for {ensemble_cls.__name__}: {exc}") from exc

    return ensemble, master_seed


def _infer_compound_type(ensemble: ManyBodyEnsemble) -> type[CompoundEnsemble]:
    if isinstance(ensemble, SYK):
        return SYKCompoundEnsemble
    if isinstance(ensemble, Poisson):
        return PoissonCompoundEnsemble
    return CompoundEnsemble


def _build_compound(
    mapping: Mapping[str, object],
    *,
    ensemble: ManyBodyEnsemble,
) -> CompoundEnsemble:
    arguments = dict(mapping)
    compound_name = _pop_name(arguments, label="Compound")
    _rename_aliases(
        arguments,
        {
            "Nf": "num_free_complex_fermions",
            "v": "couplings",
        },
    )
    if "ensemble" in arguments:
        raise ValueError("Pass the ensemble through `--ensemble`, not `--compound`.")

    if compound_name is None:
        compound_cls = _infer_compound_type(ensemble)
    else:
        try:
            compound_cls = COMPOUND_TYPES[normalize_name(compound_name)]
        except KeyError as exc:
            choices = ", ".join(("Compound", "PoissonCompound", "SYKCompound"))
            raise ValueError(
                f"Unknown compound {compound_name!r}; choose one of {choices}."
            ) from exc

    try:
        compound_factory = cast(Callable[..., CompoundEnsemble], compound_cls)
        return compound_factory(ensemble=ensemble, **arguments)
    except TypeError as exc:
        raise TypeError(f"Invalid parameters for {compound_cls.__name__}: {exc}") from exc


@attrs.frozen(kw_only=True, eq=True, weakref_slot=False)
class ScientificSpec:
    simulation: str = attrs.field(converter=normalize_name)
    ensemble: SourceDict = attrs.field(converter=deepcopy)
    compound: SourceDict | None = attrs.field(
        default=None,
        converter=attrs.converters.optional(deepcopy),
    )
    simulation_arguments: dict[str, object] = attrs.field(
        factory=dict,
        converter=deepcopy,
    )
    plot_arguments: dict[str, object] = attrs.field(
        factory=dict,
        converter=deepcopy,
    )
    realizs_per_task: int = attrs.field(
        converter=int,
        validator=attrs.validators.gt(0),
    )
    tasks: int = attrs.field(
        converter=int,
        validator=attrs.validators.gt(0),
    )
    directory: str = attrs.field(default="outputs", converter=str)
    plot: bool = attrs.field(default=True, converter=bool)
    master_seed: MasterSeed = attrs.field()

    @property
    def total_realizs(self) -> int:
        return self.realizs_per_task * self.tasks

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], attrs.asdict(self))

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, object], /) -> ScientificSpec:
        return cls(
            simulation=cast(str, value["simulation"]),
            ensemble=cast(SourceDict, value["ensemble"]),
            compound=cast(SourceDict | None, value.get("compound")),
            simulation_arguments=cast(
                dict[str, object], value.get("simulation_arguments", {})
            ),
            plot_arguments=cast(dict[str, object], value.get("plot_arguments", {})),
            realizs_per_task=cast(int, value["realizs_per_task"]),
            tasks=cast(int, value["tasks"]),
            directory=cast(str, value.get("directory", "outputs")),
            plot=cast(bool, value.get("plot", True)),
            master_seed=cast(MasterSeed, value["master_seed"]),
        )

    @classmethod
    def from_json(cls, value: str, /) -> ScientificSpec:
        decoded = cast(object, json.loads(value))
        if not isinstance(decoded, dict):
            raise TypeError("Scientific specification must be a JSON object.")
        return cls.from_dict(cast(dict[str, object], decoded))


def build_scientific_spec(
    *,
    simulation: str,
    ensemble: Mapping[str, object],
    compound: Mapping[str, object] | None,
    simulation_arguments: Mapping[str, object],
    realizs_per_task: int,
    tasks: int,
    directory: str | Path,
    plot: bool,
) -> ScientificSpec:
    simulation_name = normalize_name(simulation)
    try:
        simulation_cls = SIMULATION_TYPES[simulation_name]
    except KeyError as exc:
        choices = ", ".join(
            (
                "spectral-statistics",
                "resonance-statistics",
                "partial-widths-statistics",
                "time-delay-statistics",
                "transmission-coefficients",
            )
        )
        raise ValueError(
            f"Unknown simulation {simulation!r}; choose one of {choices}."
        ) from exc

    arguments = dict(simulation_arguments)
    allowed, required = SIMULATION_ARGUMENTS[simulation_name]
    unknown = set(arguments) - allowed
    missing = required - set(arguments)
    if unknown or missing:
        raise ValueError(
            f"Invalid arguments for {simulation_cls.__name__}: "
            + f"missing={tuple(sorted(missing))}, extra={tuple(sorted(unknown))}."
        )

    plot_arguments: dict[str, object] = {}
    if "spectral_statistics_directory" in arguments:
        plot_arguments["spectral_statistics_directory"] = arguments.pop(
            "spectral_statistics_directory"
        )

    ensemble_instance, master_seed = _build_ensemble(ensemble)
    compound_instance: CompoundEnsemble | None = None
    if simulation_cls is SpectralStatisticsSimulation:
        if compound:
            raise ValueError("Spectral statistics does not accept a compound input.")
    else:
        compound_instance = _build_compound(compound or {}, ensemble=ensemble_instance)

    spec = ScientificSpec(
        simulation=simulation_name,
        ensemble=source_dict(ensemble_instance),
        compound=(None if compound_instance is None else source_dict(compound_instance)),
        simulation_arguments=cast(dict[str, object], json_value(arguments)),
        plot_arguments=cast(dict[str, object], json_value(plot_arguments)),
        realizs_per_task=realizs_per_task,
        tasks=tasks,
        directory=str(directory),
        plot=plot,
        master_seed=master_seed,
    )

    _ = instantiate_simulation(spec, realizs=spec.total_realizs)
    return spec


def child_seed(
    spec: ScientificSpec, /, *, phase: int, rank: int
) -> np.random.SeedSequence:
    return np.random.SeedSequence(spec.master_seed, spawn_key=(phase, rank))


def _source_with_seed(
    source: SourceDict,
    seed: MasterSeed | np.random.SeedSequence,
    /,
) -> SourceDict:
    seeded_source = deepcopy(source)
    parameters = cast(dict[str, object], seeded_source["parameters"])
    parameters["seed"] = json_value(seed)
    return seeded_source


def _install_calibration(
    simulation: Simulation,
    coefficients: np.ndarray[tuple[int], np.dtype[np.floating]] | None,
) -> None:
    if coefficients is None:
        return

    if isinstance(simulation, SpectralStatisticsSimulation):
        density = simulation.ensemble.spectral_density
    elif isinstance(simulation, ResonanceStatisticsSimulation):
        density = simulation.compound.resonance_density
    elif isinstance(simulation, TimeDelayStatisticsSimulation):
        density = simulation.compound.ensemble.spectral_density
    else:
        return

    coefficients = np.asarray(
        coefficients, dtype=np.dtype(simulation.manifest.dtype["real"])
    )
    expected_shape = (density.max_polynomial_degree + 1,)
    if coefficients.shape != expected_shape or not np.all(np.isfinite(coefficients)):
        raise ValueError(
            f"Calibration coefficients must have shape {expected_shape} and be finite."
        )
    object.__setattr__(density, "average_coeffs", coefficients)


def instantiate_simulation(
    spec: ScientificSpec,
    /,
    *,
    realizs: int | None = None,
    seed: MasterSeed | np.random.SeedSequence | None = None,
    calibration: np.ndarray[tuple[int], np.dtype[np.floating]] | None = None,
) -> Simulation:
    ensemble_source = spec.ensemble
    if seed is not None:
        ensemble_source = _source_with_seed(ensemble_source, seed)
    structured_ensemble_source = unwrap_json_value(ensemble_source)
    if not isinstance(structured_ensemble_source, dict):
        raise TypeError("Ensemble source is malformed.")
    ensemble = ManyBodyEnsemble.create(cast(SourceDict, structured_ensemble_source))

    simulation_cls = SIMULATION_TYPES[spec.simulation]
    arguments = cast(
        dict[str, object], unwrap_json_value(deepcopy(spec.simulation_arguments))
    )
    arguments["realizs"] = spec.total_realizs if realizs is None else realizs

    if simulation_cls is SpectralStatisticsSimulation:
        arguments["ensemble"] = ensemble
    else:
        if spec.compound is None:
            raise ValueError(f"{simulation_cls.__name__} requires a compound input.")
        compound_source = deepcopy(spec.compound)
        compound_parameters = cast(dict[str, object], compound_source["parameters"])
        compound_parameters["ensemble"] = source_dict(ensemble)
        structured_compound_source = unwrap_json_value(compound_source)
        if not isinstance(structured_compound_source, dict):
            raise TypeError("Compound source is malformed.")
        arguments["compound"] = CompoundEnsemble.create(
            cast(SourceDict, structured_compound_source)
        )

    simulation_factory = cast(Callable[..., Simulation], simulation_cls)
    simulation = simulation_factory(**arguments)
    _install_calibration(simulation, calibration)
    return simulation


def infer_job_name(ensemble: Mapping[str, object], /) -> str:
    name = str(ensemble.get("name", ensemble.get("type", "rmtpy")))
    normalized = normalize_name(name)
    num_majoranas = ensemble.get("N", ensemble.get("num_majoranas"))
    if normalized in {"syk", "sachdevyekitaev", "sachdevyekitaevensemble"}:
        q = ensemble.get("q")
        if q is not None and num_majoranas is not None:
            return f"syk{q}_{num_majoranas}"
    if num_majoranas is not None:
        return f"{normalized}_{num_majoranas}"
    return normalized or "rmtpy"
