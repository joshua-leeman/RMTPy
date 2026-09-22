import base64
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Protocol, cast

import attrs
import numpy as np
from numpy.typing import NDArray

from ..conversion import SourceDict, get_attrs_fields
from .base_data import Data
from .base_simulation import RunContext

type ManifestMapping = dict[str, object]
type NumericArray = NDArray[np.generic]


class NumericArchive(Protocol):
    files: Sequence[str]

    def __enter__(self) -> NumericArchive: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> object: ...

    def __getitem__(self, key: str, /) -> object: ...


SCHEMA_VERSION: int = 1
MANIFEST_FILE_NAME: str = "manifest.json"
RUNTIME_DEPENDENCIES: tuple[str, ...] = (
    "attrs",
    "cattrs",
    "matplotlib",
    "numba",
    "numpy",
    "scipy",
)

TYPE_KEY: str = "__rmtpy_manifest_type__"
SAFE_TOKEN: re.Pattern[str] = re.compile(r"^[A-Za-z0-9_.-]+$")
SHA256_DIGEST: re.Pattern[str] = re.compile(r"^[0-9a-f]{64}$")
APPROVED_SOURCE_FILES: frozenset[str] = frozenset(
    {
        "environment.yml",
        "pyproject.toml",
        "requirements.txt",
        "setup.cfg",
        "setup.py",
    }
)
GIT_PATHS: tuple[str, ...] = (
    "rmtpy",
    "environment.yml",
    "pyproject.toml",
    "requirements.txt",
    "setup.cfg",
    "setup.py",
)


class PersistenceError(RuntimeError):
    pass


class RunExistsError(FileExistsError, PersistenceError):
    pass


class PersistenceSchemaError(ValueError, PersistenceError):
    pass


class PersistenceIntegrityError(ValueError, PersistenceError):
    pass


def _copy_dict[K, V](value: Mapping[K, V], /) -> dict[K, V]:
    return dict(value)


def _copy_tuple[T](value: Iterable[T], /) -> tuple[T, ...]:
    return tuple(value)


def _require_instance[T](value: object, expected_type: type[T], message: str, /) -> T:
    if not isinstance(value, expected_type):
        raise PersistenceSchemaError(message)

    return value


def _require_manifest(value: object, description: str, /) -> ManifestMapping:
    if not isinstance(value, dict):
        raise PersistenceSchemaError(f"`{description}` must be a mapping.")

    mapping: ManifestMapping = {}
    for key, item in cast(dict[object, object], value).items():
        if not isinstance(key, str):
            raise PersistenceSchemaError(f"`{description}` keys must be strings.")

        mapping[key] = item

    return mapping


def _require_list(value: object, description: str, /) -> list[object]:
    if not isinstance(value, list):
        raise PersistenceSchemaError(f"`{description}` must be a list.")

    return list(cast(list[object], value))


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class DataArtifact:
    logical_path: tuple[str, ...] = attrs.field(
        converter=_copy_tuple,
    )
    relative_path: Path = attrs.field(converter=Path)
    data: Data = attrs.field(
        validator=attrs.validators.instance_of(Data),
        repr=False,
    )
    data_type: str = attrs.field(converter=str)


@attrs.frozen(kw_only=True, eq=True, weakref_slot=False)
class DataArtifactEntry:
    run_dir: Path = attrs.field(converter=Path, repr=False)
    logical_path: tuple[str, ...] = attrs.field(
        converter=_copy_tuple,
    )
    relative_path: Path = attrs.field(converter=Path)
    data_type: str
    scalars: ManifestMapping = attrs.field(
        converter=_copy_dict,
        repr=False,
    )
    metadata: ManifestMapping = attrs.field(
        converter=_copy_dict,
        repr=False,
    )
    arrays: dict[str, ManifestMapping] = attrs.field(
        converter=_copy_dict,
        repr=False,
    )
    archive_sha256: str = attrs.field(repr=False)

    @property
    def path(self) -> Path:
        return self.run_dir / self.relative_path


def _json_value(value: object, /) -> object:
    if value is None or isinstance(value, bool | int | str):
        return value

    if isinstance(value, float):
        if math.isfinite(value):
            return value

        if math.isnan(value):
            token = "nan"
        elif value > 0:
            token = "inf"
        else:
            token = "-inf"

        return {TYPE_KEY: "float", "value": token}

    if isinstance(value, complex):
        return {
            TYPE_KEY: "complex",
            "real": _json_value(value.real),
            "imag": _json_value(value.imag),
        }

    if isinstance(value, bytes):
        return {
            TYPE_KEY: "bytes",
            "base64": base64.b64encode(value).decode("ascii"),
        }

    if isinstance(value, np.dtype):
        return {TYPE_KEY: "dtype", "value": value.str}

    if isinstance(value, np.generic):
        return _json_value(value.item())

    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object arrays are not supported by persistence schema v1.")

        return {
            TYPE_KEY: "ndarray",
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "values": _json_value(cast(list[object], value.tolist())),
        }

    if isinstance(value, tuple):
        items = cast(tuple[object, ...], value)
        return {TYPE_KEY: "tuple", "items": [_json_value(item) for item in items]}

    if isinstance(value, list):
        items = cast(list[object], value)
        return [_json_value(item) for item in items]

    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        normalized: ManifestMapping = {}
        for key, item in mapping.items():
            if not isinstance(key, str):
                raise TypeError("Manifest mappings must use string keys.")
            if key == TYPE_KEY:
                raise ValueError(f"{TYPE_KEY!r} is reserved by persistence schema v1.")

            normalized[key] = _json_value(item)

        return normalized

    raise TypeError(f"Unsupported manifest value: {type(value).__name__}.")


def _decoded_json_value(value: object, /) -> object:
    if isinstance(value, list):
        items = _require_list(cast(list[object], value), "encoded list")
        return [_decoded_json_value(item) for item in items]

    if not isinstance(value, dict):
        return value

    mapping = _require_manifest(cast(dict[object, object], value), "encoded value")
    if TYPE_KEY not in mapping:
        return {key: _decoded_json_value(item) for key, item in mapping.items()}

    token = mapping[TYPE_KEY]
    if token == "float":
        _require_exact_keys(mapping, {TYPE_KEY, "value"}, "encoded float")
        encoded_float = _require_instance(
            mapping["value"],
            str,
            "`encoded float` value must be a string.",
        )
        return {
            "nan": math.nan,
            "inf": math.inf,
            "-inf": -math.inf,
        }[encoded_float]

    if token == "complex":
        _require_exact_keys(mapping, {TYPE_KEY, "real", "imag"}, "encoded complex")
        real = _decoded_json_value(mapping["real"])
        imag = _decoded_json_value(mapping["imag"])
        if (
            not isinstance(real, int | float)
            or isinstance(real, bool)
            or not isinstance(imag, int | float)
            or isinstance(imag, bool)
        ):
            raise PersistenceSchemaError(
                "`encoded complex` components must be real numbers."
            )

        return complex(real, imag)

    if token == "bytes":
        _require_exact_keys(mapping, {TYPE_KEY, "base64"}, "encoded bytes")
        encoded_bytes = _require_instance(
            mapping["base64"],
            str,
            "`encoded bytes` base64 value must be a string.",
        )
        try:
            return base64.b64decode(encoded_bytes, validate=True)
        except (TypeError, ValueError) as exc:
            raise PersistenceSchemaError("Invalid base64 bytes value.") from exc

    if token == "dtype":
        _require_exact_keys(mapping, {TYPE_KEY, "value"}, "encoded dtype")
        encoded_dtype = _require_instance(
            mapping["value"],
            str,
            "`encoded dtype` value must be a string.",
        )
        try:
            return np.dtype(encoded_dtype)
        except TypeError as exc:
            raise PersistenceSchemaError("Invalid dtype value.") from exc

    if token == "tuple":
        _require_exact_keys(mapping, {TYPE_KEY, "items"}, "encoded tuple")
        encoded_items = _require_list(
            mapping["items"],
            "encoded tuple items",
        )
        return tuple(_decoded_json_value(item) for item in encoded_items)

    if token == "ndarray":
        _require_exact_keys(
            mapping,
            {TYPE_KEY, "dtype", "shape", "values"},
            "encoded ndarray",
        )
        shape = _validate_shape(mapping["shape"], "encoded ndarray")
        dtype = _require_instance(
            mapping["dtype"],
            str,
            "`encoded ndarray` dtype must be a string.",
        )
        try:
            array = np.asarray(
                _decoded_json_value(mapping["values"]),
                dtype=dtype,
            )
        except (TypeError, ValueError) as exc:
            raise PersistenceSchemaError("Invalid encoded ndarray values.") from exc

        if array.shape != shape:
            raise PersistenceSchemaError(
                f"Encoded ndarray has shape {array.shape}, expected {shape}."
            )

        return array

    raise PersistenceSchemaError(f"Unknown encoded value type {token!r}.")


def _canonical_json(value: object, /) -> bytes:
    return json.dumps(
        _json_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _require_exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    description: str,
    /,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = tuple(sorted(expected - actual))
        extra = tuple(sorted(actual - expected))
        raise PersistenceSchemaError(
            f"Invalid {description} keys; missing={missing}, extra={extra}."
        )


def _validate_token(value: object, description: str, /) -> str:
    if not isinstance(value, str) or SAFE_TOKEN.fullmatch(value) is None:
        raise PersistenceSchemaError(
            f"`{description}` must contain only letters, digits, '.', '_', and '-'."
        )

    return value


def _validate_sha256(value: object, description: str, /) -> str:
    if not isinstance(value, str) or SHA256_DIGEST.fullmatch(value) is None:
        raise PersistenceSchemaError(f"`{description}` must be a full SHA256 digest.")

    return value


def _validate_shape(value: object, description: str, /) -> tuple[int, ...]:
    items = _require_list(value, f"{description} shape")
    shape: list[int] = []
    for item in items:
        if type(item) is not int or item < 0:
            raise PersistenceSchemaError(
                f"`{description}` shape must contain nonnegative integers."
            )

        shape.append(item)

    return tuple(shape)


def _validate_logical_path(value: object, /) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise PersistenceSchemaError(
            "Artifact `logical_path` must be a sequence of tokens."
        )

    logical_path = tuple(value)
    if not logical_path:
        raise PersistenceSchemaError("Artifact `logical_path` must not be empty.")

    return tuple(
        _validate_token(token, "Artifact logical_path token") for token in logical_path
    )


def _validate_relative_path(value: str | Path, /) -> Path:
    path = Path(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.suffix != ".npz"
    ):
        raise PersistenceSchemaError(
            "Artifact `relative_path` must be a normalized relative .npz path."
        )

    return path


def _array_sha256(array: NumericArray, /) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json(list(contiguous.shape)))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _file_sha256(path: Path, /) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_numeric_array(array: NumericArray, /) -> bool:
    return not array.dtype.hasobject and (
        np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)
    )


def _split_data_fields(
    data: Data,
    /,
) -> tuple[ManifestMapping, dict[str, NumericArray]]:
    if not attrs.has(type(data)):
        raise TypeError(
            f"Persisted data must be an attrs class, got {type(data).__name__}."
        )

    scalars: ManifestMapping = {}
    arrays: dict[str, NumericArray] = {}
    fields = get_attrs_fields(type(data))
    for field in fields:
        if field.name == "metadata":
            continue

        value = cast(object, getattr(data, field.name))
        if isinstance(value, np.ndarray):
            array = np.asarray(value)
            if not _is_numeric_array(array):
                raise TypeError(
                    f"Data array {field.name!r} must have a numeric or boolean dtype."
                )

            arrays[field.name] = array
        else:
            scalars[field.name] = _json_value(value)

    return scalars, arrays


def _approved_source_path(value: str, /) -> bool:
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return False
    posix = path.as_posix()
    if posix in APPROVED_SOURCE_FILES:
        return True
    return len(path.parts) > 1 and path.parts[0] == "rmtpy" and path.suffix == ".py"


def _git(
    repository_root: Path,
    /,
    *arguments: str,
) -> subprocess.CompletedProcess[bytes] | None:
    try:
        return subprocess.run(
            ("git", *arguments),
            cwd=repository_root,
            check=False,
            capture_output=True,
        )
    except OSError:
        return None


def _nul_paths(output: bytes, /) -> tuple[str, ...]:
    return tuple(
        item.decode("utf-8", errors="surrogateescape")
        for item in output.split(b"\0")
        if item
    )


def _source_paths(repository_root: Path, /, *, git_available: bool) -> tuple[str, ...]:
    paths: set[str] = set()
    if git_available:
        result = _git(
            repository_root,
            "ls-files",
            "-co",
            "--exclude-standard",
            "-z",
            "--",
            *GIT_PATHS,
        )
        if result is not None and result.returncode == 0:
            paths.update(_nul_paths(result.stdout))
    else:
        package_root = repository_root / "rmtpy"
        if package_root.is_dir():
            paths.update(
                path.relative_to(repository_root).as_posix()
                for path in package_root.rglob("*.py")
                if path.is_file()
            )
        paths.update(
            name for name in APPROVED_SOURCE_FILES if (repository_root / name).is_file()
        )
    return tuple(sorted(path for path in paths if _approved_source_path(path)))


def _source_digest(repository_root: Path, paths: Sequence[str], /) -> str:
    digest = hashlib.sha256()
    for relative in paths:
        path = repository_root / relative
        if not path.is_file() or path.is_symlink():
            continue
        digest.update(relative.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _context_payload(
    context: RunContext,
    /,
    *,
    include_final_rng_state: bool,
) -> ManifestMapping:
    rng = dict(context.rng)
    if not include_final_rng_state:
        _ = rng.pop("final_state", None)

    return {
        "simulation_type": context.simulation_type,
        "result_type": context.result_type,
        "simulation_config": _json_value(context.simulation_config),
        "output_request": _json_value(context.output_request),
        "rng": _json_value(rng),
        "dtype": _json_value(context.dtype),
        "execution": _json_value(context.execution),
    }


def _validate_software(software: object, /) -> ManifestMapping:
    software_mapping = _require_manifest(software, "Manifest software")
    _require_exact_keys(
        software_mapping,
        {"code_version", "python", "dependencies", "repository"},
        "software",
    )

    python_mapping = _require_manifest(
        software_mapping["python"],
        "Manifest Python metadata",
    )
    _require_exact_keys(python_mapping, {"implementation", "version"}, "Python metadata")
    if any(
        not isinstance(python_mapping[key], str) or not python_mapping[key]
        for key in python_mapping
    ):
        raise PersistenceSchemaError("Manifest Python metadata values must be strings.")

    dependencies_mapping = _require_manifest(
        software_mapping["dependencies"],
        "Manifest dependencies",
    )
    if set(dependencies_mapping) != set(RUNTIME_DEPENDENCIES):
        raise PersistenceSchemaError("Manifest dependencies do not match schema v1.")
    if any(
        version is not None and (not isinstance(version, str) or not version)
        for version in dependencies_mapping.values()
    ):
        raise PersistenceSchemaError(
            "Manifest dependency versions must be strings or null."
        )

    repository_mapping = _require_manifest(
        software_mapping["repository"],
        "Manifest repository metadata",
    )
    _require_exact_keys(
        repository_mapping,
        {
            "git_available",
            "commit",
            "dirty",
            "dirty_paths",
            "dirty_digest",
            "source_digest",
        },
        "repository metadata",
    )
    _ = _validate_sha256(repository_mapping["source_digest"], "Repository source_digest")

    git_available = _require_instance(
        repository_mapping["git_available"],
        bool,
        "`Repository git_available` must be boolean.",
    )
    dirty_path_items = _require_list(
        repository_mapping["dirty_paths"],
        "Repository dirty_paths",
    )
    dirty_path_tokens: list[str] = []
    for path in dirty_path_items:
        if not isinstance(path, str) or not _approved_source_path(path):
            raise PersistenceSchemaError(
                "`Repository dirty_paths` contains an absolute or unapproved path."
            )

        dirty_path_tokens.append(path)

    if dirty_path_tokens != sorted(set(dirty_path_tokens)):
        raise PersistenceSchemaError(
            "`Repository dirty_paths` must be sorted and unique."
        )

    dirty_digest = repository_mapping["dirty_digest"]
    if dirty_digest is not None:
        dirty_digest = _validate_sha256(dirty_digest, "Repository dirty_digest")

    commit = repository_mapping["commit"]
    dirty = repository_mapping["dirty"]
    if git_available:
        if (
            not isinstance(commit, str)
            or re.fullmatch(r"[0-9a-f]{40,64}", commit) is None
            or type(dirty) is not bool
        ):
            raise PersistenceSchemaError("Repository Git state is malformed.")

        if dirty:
            if not dirty_path_tokens or dirty_digest is None:
                raise PersistenceSchemaError("Dirty repository details are incomplete.")
        elif dirty_path_tokens or dirty_digest is not None:
            raise PersistenceSchemaError("Clean repository details are inconsistent.")
    elif (
        commit is not None
        or dirty is not None
        or dirty_digest is not None
        or dirty_path_tokens
    ):
        raise PersistenceSchemaError("Non-Git repository details are inconsistent.")

    code_version = _require_instance(
        software_mapping["code_version"],
        str,
        "`Manifest code_version` must be a string.",
    )
    if not code_version:
        raise PersistenceSchemaError("`Manifest code_version` must not be empty.")

    if commit is None:
        source_digest = _validate_sha256(
            repository_mapping["source_digest"],
            "Repository source_digest",
        )
        expected_version = f"source-{source_digest}"
    elif dirty:
        expected_version = f"{commit}+dirty.{dirty_digest}"
    else:
        expected_version = commit

    if code_version != expected_version:
        raise PersistenceSchemaError("`Manifest code_version` is inconsistent.")

    return software_mapping


def _validate_decoded_context(value: object, /) -> ManifestMapping:
    context = _require_manifest(value, "Decoded run context")
    _require_exact_keys(
        context,
        {
            "simulation_type",
            "result_type",
            "simulation_config",
            "output_request",
            "rng",
            "dtype",
            "execution",
        },
        "decoded run context",
    )
    _ = _validate_token(context["simulation_type"], "simulation_type")
    _ = _validate_token(context["result_type"], "result_type")

    configuration_mapping = _require_manifest(
        context["simulation_config"],
        "Simulation configuration",
    )
    _require_exact_keys(
        configuration_mapping, {"type", "parameters"}, "simulation config"
    )
    _ = _validate_token(configuration_mapping["type"], "Simulation configuration type")
    _ = _require_manifest(
        configuration_mapping["parameters"],
        "Simulation parameters",
    )
    _ = _require_manifest(context["output_request"], "Output request")
    _ = _require_manifest(context["execution"], "Execution information")

    rng_mapping = _require_manifest(context["rng"], "RNG information")
    _require_exact_keys(
        rng_mapping,
        {
            "policy",
            "bit_generator",
            "seed",
            "state_policy",
            "initial_state",
            "final_state",
        },
        "RNG information",
    )
    if rng_mapping["policy"] != "numpy.random.default_rng":
        raise PersistenceSchemaError("Unsupported RNG policy.")
    _ = _validate_token(rng_mapping["bit_generator"], "RNG bit generator")
    if rng_mapping["state_policy"] != "capture_initial_and_final":
        raise PersistenceSchemaError("Unsupported RNG state policy.")
    if not isinstance(rng_mapping["initial_state"], dict) or not isinstance(
        rng_mapping["final_state"], dict
    ):
        raise PersistenceSchemaError("RNG states must be mappings.")

    dtype_mapping = _require_manifest(context["dtype"], "Dtype information")
    _require_exact_keys(
        dtype_mapping, {"configured", "real", "complex"}, "dtype information"
    )
    configured_name = _require_instance(
        dtype_mapping["configured"],
        str,
        "`Dtype configured` must be a string.",
    )
    real_name = _require_instance(
        dtype_mapping["real"],
        str,
        "`Dtype real` must be a string.",
    )
    complex_name = _require_instance(
        dtype_mapping["complex"],
        str,
        "`Dtype complex` must be a string.",
    )
    try:
        configured = np.dtype(configured_name)
        real = np.dtype(real_name)
        complex_dtype = np.dtype(complex_name)
    except (TypeError, ValueError) as exc:
        raise PersistenceSchemaError("Dtype information is malformed.") from exc

    if configured.hasobject or real.kind != "f" or complex_dtype.kind != "c":
        raise PersistenceSchemaError("Dtype roles are malformed.")

    return context


def _write_npz(path: Path, arrays: Mapping[str, NumericArray], /) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as file:
        save_archive = cast(Callable[..., None], np.savez)
        save_archive(file, **dict(arrays))
        file.flush()
        os.fsync(file.fileno())


def _write_manifest(path: Path, manifest: Mapping[str, object], /) -> None:
    encoded = json.dumps(
        dict(manifest),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    with path.open("x", encoding="utf-8") as file:
        _ = file.write(encoded)
        _ = file.write("\n")
        file.flush()
        os.fsync(file.fileno())


def _fsync_directory(path: Path, /) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _artifact_manifest_entry(
    artifact: DataArtifact,
    /,
    *,
    staging_dir: Path,
) -> ManifestMapping:
    logical_path = _validate_logical_path(artifact.logical_path)
    relative_path = _validate_relative_path(artifact.relative_path)
    data_type = _validate_token(artifact.data_type, "Artifact data_type")
    scalars, arrays = _split_data_fields(artifact.data)

    archive_path = staging_dir / relative_path
    _write_npz(archive_path, arrays)
    array_entries = {
        name: {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": _array_sha256(array),
        }
        for name, array in arrays.items()
    }
    return {
        "logical_path": list(logical_path),
        "relative_path": relative_path.as_posix(),
        "data_type": data_type,
        "scalars": scalars,
        "metadata": _json_value(artifact.data.metadata),
        "arrays": array_entries,
        "archive_sha256": _file_sha256(archive_path),
    }


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
    /,
) -> ManifestMapping:
    result: ManifestMapping = {}
    for key, value in pairs:
        if key in result:
            raise PersistenceSchemaError(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _read_manifest(run_dir: Path, /) -> ManifestMapping:
    path = run_dir / MANIFEST_FILE_NAME
    if not path.is_file() or path.is_symlink():
        raise PersistenceSchemaError(
            f"{run_dir} is not a schema-v{SCHEMA_VERSION} saved run."
        )
    try:
        with path.open(encoding="utf-8") as file:
            value = cast(
                object,
                json.load(
                    file,
                    object_pairs_hook=_reject_duplicate_json_keys,
                    parse_constant=lambda token: (_ for _ in ()).throw(
                        PersistenceSchemaError(f"Invalid JSON constant {token!r}.")
                    ),
                ),
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PersistenceSchemaError("Could not read the schema-v1 manifest.") from exc

    return _require_manifest(value, "Run manifest")


def _parse_artifact_entry(run_dir: Path, value: object, /) -> DataArtifactEntry:
    entry_value = _require_manifest(value, "Manifest content entry")
    _require_exact_keys(
        entry_value,
        {
            "logical_path",
            "relative_path",
            "data_type",
            "scalars",
            "metadata",
            "arrays",
            "archive_sha256",
        },
        "content entry",
    )

    logical_path = _validate_logical_path(entry_value["logical_path"])
    relative_path = _validate_relative_path(
        _require_instance(
            entry_value["relative_path"],
            str,
            "`Artifact relative_path` must be a string.",
        )
    )
    data_type = _validate_token(entry_value["data_type"], "Artifact data_type")
    scalars = _require_manifest(entry_value["scalars"], "Artifact scalars")
    metadata = _require_manifest(entry_value["metadata"], "Artifact metadata")
    array_mapping = _require_manifest(entry_value["arrays"], "Artifact arrays")

    arrays: dict[str, ManifestMapping] = {}
    for name, descriptor in array_mapping.items():
        descriptor_mapping = _require_manifest(
            descriptor,
            f"Artifact array descriptor {name!r}",
        )
        _require_exact_keys(
            descriptor_mapping,
            {"dtype", "shape", "sha256"},
            f"array descriptor {name!r}",
        )
        dtype_name = _require_instance(
            descriptor_mapping["dtype"],
            str,
            f"`Array {name!r} dtype` must be a string.",
        )
        try:
            dtype = np.dtype(dtype_name)
        except TypeError as exc:
            raise PersistenceSchemaError(f"Invalid dtype for array {name!r}.") from exc

        if dtype.hasobject or not _is_numeric_array(np.empty(0, dtype=dtype)):
            raise PersistenceSchemaError(f"Array {name!r} must have a numeric dtype.")

        _ = _validate_shape(descriptor_mapping["shape"], f"Array {name!r}")
        _ = _validate_sha256(descriptor_mapping["sha256"], f"Array {name!r} SHA256")
        arrays[name] = descriptor_mapping

    archive_sha256 = _validate_sha256(
        entry_value["archive_sha256"],
        "Artifact archive SHA256",
    )
    entry = DataArtifactEntry(
        run_dir=run_dir,
        logical_path=logical_path,
        relative_path=relative_path,
        data_type=data_type,
        scalars=scalars,
        metadata=metadata,
        arrays=arrays,
        archive_sha256=archive_sha256,
    )
    if not entry.path.is_file() or entry.path.is_symlink():
        raise PersistenceIntegrityError(f"Missing data artifact: {relative_path}")
    if _file_sha256(entry.path) != archive_sha256:
        raise PersistenceIntegrityError(f"Artifact checksum failed: {relative_path}")

    return entry


def _build_run_context(decoded_context: ManifestMapping, /) -> RunContext:
    configuration = _require_manifest(
        decoded_context["simulation_config"],
        "Simulation configuration",
    )
    simulation_config: SourceDict = {
        "type": _validate_token(
            configuration["type"],
            "Simulation configuration type",
        ),
        "parameters": _require_manifest(
            configuration["parameters"],
            "Simulation parameters",
        ),
    }
    dtype_mapping = _require_manifest(decoded_context["dtype"], "Dtype information")
    dtype = {
        name: _require_instance(
            dtype_mapping[name],
            str,
            f"`Dtype {name}` must be a string.",
        )
        for name in ("configured", "real", "complex")
    }

    return RunContext(
        simulation_type=_validate_token(
            decoded_context["simulation_type"],
            "simulation_type",
        ),
        result_type=_validate_token(
            decoded_context["result_type"],
            "result_type",
        ),
        simulation_config=simulation_config,
        output_request=_require_manifest(
            decoded_context["output_request"],
            "Output request",
        ),
        rng=_require_manifest(decoded_context["rng"], "RNG information"),
        dtype=dtype,
        execution=_require_manifest(
            decoded_context["execution"],
            "Execution information",
        ),
    )


def collect_repository_provenance(
    repository_root: str | Path | None = None,
    /,
) -> ManifestMapping:
    root = (
        Path(repository_root)
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    probe = _git(root, "rev-parse", "--is-inside-work-tree")
    git_available = (
        probe is not None and probe.returncode == 0 and probe.stdout.strip() == b"true"
    )
    paths = _source_paths(root, git_available=git_available)
    source_digest = _source_digest(root, paths)

    if not git_available:
        return {
            "git_available": False,
            "commit": None,
            "dirty": None,
            "dirty_paths": [],
            "dirty_digest": None,
            "source_digest": source_digest,
        }

    commit_result = _git(root, "rev-parse", "HEAD")
    if commit_result is None or commit_result.returncode != 0:
        raise PersistenceError("Could not read the repository commit.")
    commit = commit_result.stdout.decode("ascii").strip()

    changed_result = _git(
        root,
        "diff",
        "--name-only",
        "-z",
        "HEAD",
        "--",
        *GIT_PATHS,
    )
    untracked_result = _git(
        root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
        *GIT_PATHS,
    )
    if (
        changed_result is None
        or changed_result.returncode != 0
        or untracked_result is None
        or untracked_result.returncode != 0
    ):
        raise PersistenceError("Could not inspect the repository working tree.")

    changed = set(_nul_paths(changed_result.stdout))
    untracked = set(_nul_paths(untracked_result.stdout))
    dirty_paths = tuple(
        sorted(path for path in changed | untracked if _approved_source_path(path))
    )

    dirty_digest = None
    if dirty_paths:
        tracked_dirty_paths = tuple(sorted(changed & set(dirty_paths)))
        digest = hashlib.sha256()
        if tracked_dirty_paths:
            patch_result = _git(
                root,
                "diff",
                "--binary",
                "HEAD",
                "--",
                *tracked_dirty_paths,
            )
            if patch_result is None or patch_result.returncode != 0:
                raise PersistenceError("Could not inspect repository source changes.")
            digest.update(patch_result.stdout)
        for relative in sorted(untracked):
            if not _approved_source_path(relative):
                continue
            path = root / relative
            digest.update(relative.encode("utf-8", errors="surrogateescape"))
            digest.update(b"\0")
            if path.is_file() and not path.is_symlink():
                with path.open("rb") as file:
                    while chunk := file.read(1024 * 1024):
                        digest.update(chunk)
            digest.update(b"\0")
        dirty_digest = digest.hexdigest()

    return {
        "git_available": True,
        "commit": commit,
        "dirty": bool(dirty_paths),
        "dirty_paths": list(dirty_paths),
        "dirty_digest": dirty_digest,
        "source_digest": source_digest,
    }


def collect_software_metadata(
    repository_root: str | Path | None = None,
    /,
) -> ManifestMapping:
    repository = collect_repository_provenance(repository_root)
    if repository["commit"] is None:
        code_version = f"source-{repository['source_digest']}"
    elif repository["dirty"]:
        code_version = f"{repository['commit']}+dirty.{repository['dirty_digest']}"
    else:
        code_version = _require_instance(
            repository["commit"],
            str,
            "`Repository commit` must be a string.",
        )

    dependencies: dict[str, str | None] = {}
    for name in RUNTIME_DEPENDENCIES:
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None

    return {
        "code_version": code_version,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "dependencies": dependencies,
        "repository": repository,
    }


def compute_run_id(
    context: RunContext,
    /,
    *,
    software: Mapping[str, object] | None = None,
) -> str:
    if software is None:
        software = collect_software_metadata()
    _ = _validate_decoded_context(
        _decoded_json_value(_context_payload(context, include_final_rng_state=True))
    )
    _ = _validate_software(_decoded_json_value(_json_value(software)))
    identity = {
        "schema_version": SCHEMA_VERSION,
        "context": _context_payload(context, include_final_rng_state=False),
        "software": _json_value(software),
    }
    return hashlib.sha256(_canonical_json(identity)).hexdigest()


def save_run(
    context: RunContext,
    artifacts: Sequence[DataArtifact],
    /,
    *,
    out_dir: str | Path,
) -> Path:
    _ = _validate_decoded_context(
        _decoded_json_value(_context_payload(context, include_final_rng_state=True))
    )
    simulation_type = _validate_token(context.simulation_type, "simulation_type")
    _ = _validate_token(context.result_type, "result_type")
    artifacts = tuple(artifacts)
    if not artifacts:
        raise PersistenceSchemaError(
            "A saved run must contain at least one data artifact."
        )

    logical_paths = tuple(_validate_logical_path(item.logical_path) for item in artifacts)
    relative_paths = tuple(
        _validate_relative_path(item.relative_path) for item in artifacts
    )
    if len(set(logical_paths)) != len(logical_paths):
        raise PersistenceSchemaError("Artifact logical paths must be unique.")
    if len(set(relative_paths)) != len(relative_paths):
        raise PersistenceSchemaError("Artifact relative paths must be unique.")

    for item in artifacts:
        _ = _validate_token(item.data_type, "Artifact data_type")
        _ = _split_data_fields(item.data)
        _ = _json_value(item.data.metadata)

    software = collect_software_metadata()
    run_id = compute_run_id(context, software=software)
    family_dir = Path(out_dir) / simulation_type
    family_dir.mkdir(parents=True, exist_ok=True)
    final_dir = family_dir / run_id
    lock_dir = family_dir / f".{run_id}.lock"
    partial_pattern = f".{run_id}.partial-*"

    if final_dir.exists() or final_dir.is_symlink():
        raise RunExistsError(f"Run destination already exists: {final_dir}")
    if lock_dir.exists() or any(family_dir.glob(partial_pattern)):
        raise RunExistsError(f"A partial or locked run already exists for {run_id}.")
    try:
        lock_dir.mkdir()
    except FileExistsError as exc:
        raise RunExistsError(
            f"A run writer already holds the lock for {run_id}."
        ) from exc

    staging_dir: Path | None = None
    published = False
    try:
        if (
            final_dir.exists()
            or final_dir.is_symlink()
            or any(family_dir.glob(partial_pattern))
        ):
            raise RunExistsError(f"Run destination became unavailable for {run_id}.")

        staging_dir = Path(tempfile.mkdtemp(prefix=f".{run_id}.partial-", dir=family_dir))
        content_entries = tuple(
            _artifact_manifest_entry(artifact, staging_dir=staging_dir)
            for artifact in sorted(
                artifacts,
                key=lambda item: (item.logical_path, item.relative_path.as_posix()),
            )
        )
        manifest: ManifestMapping = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "context": _context_payload(context, include_final_rng_state=True),
            "software": _json_value(software),
            "contents": list(content_entries),
        }
        _write_manifest(staging_dir / MANIFEST_FILE_NAME, manifest)
        _fsync_directory(staging_dir)

        if final_dir.exists() or final_dir.is_symlink():
            raise RunExistsError(f"Run destination already exists: {final_dir}")

        os.rename(staging_dir, final_dir)
        published = True
        _fsync_directory(family_dir)
    finally:
        if not published and staging_dir is not None and staging_dir.exists():
            shutil.rmtree(staging_dir)
        with suppress(FileNotFoundError):
            lock_dir.rmdir()
        _fsync_directory(family_dir)

    return final_dir


def load_run(
    run_dir: str | Path,
    /,
    *,
    expected_simulation_type: str,
    expected_result_type: str,
) -> tuple[RunContext, tuple[DataArtifactEntry, ...]]:
    run_dir = Path(run_dir)
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise PersistenceSchemaError(
            f"{run_dir} is not a schema-v{SCHEMA_VERSION} saved run directory."
        )

    manifest = _read_manifest(run_dir)
    _require_exact_keys(
        manifest,
        {"schema_version", "run_id", "context", "software", "contents"},
        "manifest",
    )
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != SCHEMA_VERSION
    ):
        raise PersistenceSchemaError(
            f"Unsupported persistence schema {manifest['schema_version']!r}; "
            + f"expected {SCHEMA_VERSION}."
        )
    run_id = _validate_sha256(manifest["run_id"], "run_id")
    if run_dir.name != run_id:
        raise PersistenceIntegrityError("Run directory name does not match run_id.")

    context_mapping = _require_manifest(manifest["context"], "Manifest context")
    _require_exact_keys(
        context_mapping,
        {
            "simulation_type",
            "result_type",
            "simulation_config",
            "output_request",
            "rng",
            "dtype",
            "execution",
        },
        "run context",
    )
    simulation_type = _validate_token(
        context_mapping["simulation_type"], "simulation_type"
    )
    result_type = _validate_token(context_mapping["result_type"], "result_type")
    if simulation_type != expected_simulation_type:
        raise PersistenceSchemaError(
            f"Expected simulation type {expected_simulation_type!r}, "
            + f"got {simulation_type!r}."
        )
    if result_type != expected_result_type:
        raise PersistenceSchemaError(
            f"Expected result type {expected_result_type!r}, got {result_type!r}."
        )
    if run_dir.parent.name != simulation_type:
        raise PersistenceIntegrityError(
            "Run parent directory does not match the manifest simulation type."
        )
    decoded_context = _validate_decoded_context(_decoded_json_value(context_mapping))
    try:
        context = _build_run_context(decoded_context)
    except (TypeError, ValueError) as exc:
        raise PersistenceSchemaError("Invalid run context values.") from exc

    software = _validate_software(manifest["software"])
    if compute_run_id(context, software=software) != run_id:
        raise PersistenceIntegrityError(
            "Manifest reproduction inputs do not match run_id."
        )

    contents = _require_list(manifest["contents"], "Manifest contents")
    if not contents:
        raise PersistenceSchemaError("Manifest contents must be a nonempty list.")
    entries = tuple(_parse_artifact_entry(run_dir, item) for item in contents)
    logical_paths = tuple(entry.logical_path for entry in entries)
    relative_paths = tuple(entry.relative_path for entry in entries)
    if len(set(logical_paths)) != len(logical_paths):
        raise PersistenceSchemaError("Manifest logical paths must be unique.")
    if len(set(relative_paths)) != len(relative_paths):
        raise PersistenceSchemaError("Manifest relative paths must be unique.")

    expected_files = {MANIFEST_FILE_NAME, *(path.as_posix() for path in relative_paths)}
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for path in run_dir.rglob("*"):
        if path.is_symlink():
            raise PersistenceIntegrityError("Saved runs must not contain symbolic links.")
        if path.is_file():
            actual_files.add(path.relative_to(run_dir).as_posix())
        elif path.is_dir():
            actual_directories.add(path.relative_to(run_dir).as_posix())
    if actual_files != expected_files:
        missing = tuple(sorted(expected_files - actual_files))
        extra_files = tuple(sorted(actual_files - expected_files))
        raise PersistenceIntegrityError(
            "Saved run files do not match the manifest; "
            + f"missing={missing}, extra={extra_files}."
        )
    expected_directories = {
        parent.as_posix()
        for relative_path in relative_paths
        for parent in relative_path.parents
        if parent != Path(".")
    }
    if actual_directories != expected_directories:
        missing = tuple(sorted(expected_directories - actual_directories))
        extra_directories = tuple(sorted(actual_directories - expected_directories))
        raise PersistenceIntegrityError(
            "Saved run directories do not match the manifest; "
            + f"missing={missing}, extra={extra_directories}."
        )

    return context, entries


def load_data_artifact(
    entry: DataArtifactEntry,
    /,
    *,
    expected_type: type[Data],
    expected_token: str,
) -> Data:
    expected_token = _validate_token(expected_token, "expected_token")
    if entry.data_type != expected_token:
        raise PersistenceSchemaError(
            f"Expected data token {expected_token!r}, got {entry.data_type!r}."
        )

    fields = tuple(
        field for field in get_attrs_fields(expected_type) if field.name != "metadata"
    )
    non_init_fields = tuple(field.name for field in fields if not field.init)
    if non_init_fields:
        raise PersistenceSchemaError(
            f"{expected_type.__name__} has unsupported hidden persisted fields "
            + f"{non_init_fields}."
        )
    expected_fields = {field.name for field in fields}
    stored_fields = set(entry.scalars) | set(entry.arrays)
    if set(entry.scalars) & set(entry.arrays) or stored_fields != expected_fields:
        missing = tuple(sorted(expected_fields - stored_fields))
        extra_fields = tuple(sorted(stored_fields - expected_fields))
        raise PersistenceSchemaError(
            f"Stored {expected_type.__name__} fields do not match; "
            + f"missing={missing}, extra={extra_fields}."
        )

    arrays: dict[str, NumericArray] = {}
    try:
        with cast(NumericArchive, np.load(entry.path, allow_pickle=False)) as archive:
            if set(archive.files) != set(entry.arrays):
                raise PersistenceIntegrityError(
                    "NPZ members do not match the artifact manifest."
                )

            for name, descriptor in entry.arrays.items():
                array = cast(NumericArray, np.array(archive[name], copy=True))
                expected_dtype = np.dtype(
                    _require_instance(
                        descriptor["dtype"],
                        str,
                        f"`Array {name!r} dtype` must be a string.",
                    )
                )
                expected_shape = _validate_shape(
                    descriptor["shape"],
                    f"Array {name!r}",
                )
                if array.dtype != expected_dtype or array.shape != expected_shape:
                    raise PersistenceIntegrityError(
                        f"Array {name!r} dtype or shape does not match its manifest."
                    )

                if not _is_numeric_array(array):
                    raise PersistenceIntegrityError(
                        f"Array {name!r} is not numeric or boolean."
                    )

                if _array_sha256(array) != descriptor["sha256"]:
                    raise PersistenceIntegrityError(
                        f"Array {name!r} checksum does not match its manifest."
                    )

                arrays[name] = array
    except (OSError, ValueError) as exc:
        if isinstance(exc, PersistenceError):
            raise
        raise PersistenceIntegrityError(
            "Could not load the numeric NPZ artifact."
        ) from exc

    scalars = {name: _decoded_json_value(value) for name, value in entry.scalars.items()}
    values: ManifestMapping = {**scalars, **arrays}
    init_values = {field.name: values[field.name] for field in fields}
    try:
        metadata = _decoded_json_value(entry.metadata)
        data = expected_type(
            metadata=_require_manifest(
                metadata,
                "Decoded artifact metadata",
            ),
            **init_values,
        )
        attrs.validate(data)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, PersistenceError):
            raise
        raise PersistenceSchemaError(
            f"Could not construct {expected_type.__name__} from the current schema."
        ) from exc

    return data


__all__ = [
    "DataArtifact",
    "DataArtifactEntry",
    "MANIFEST_FILE_NAME",
    "RUNTIME_DEPENDENCIES",
    "PersistenceError",
    "PersistenceIntegrityError",
    "PersistenceSchemaError",
    "RunExistsError",
    "SCHEMA_VERSION",
    "collect_repository_provenance",
    "collect_software_metadata",
    "compute_run_id",
    "load_data_artifact",
    "load_run",
    "save_run",
]
