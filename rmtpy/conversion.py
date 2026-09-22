import hashlib
import math
import re
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import cast

import attrs
import cattrs
import numpy as np
from numpy.typing import DTypeLike, NDArray

type SourceDict = dict[str, str | dict[str, object]]
type AttrsField = attrs.Attribute[object]
type AttrsFields = dict[str, AttrsField]

RMT_CONVERTER: cattrs.Converter = cattrs.Converter()


def _structure_dtype(value: DTypeLike, _: object) -> np.dtype[np.generic]:
    return np.dtype(value)


def _unstructure_dtype(dtype: np.dtype[np.generic]) -> str:
    return dtype.name


RMT_CONVERTER.register_structure_hook(np.dtype, _structure_dtype)
RMT_CONVERTER.register_unstructure_hook(np.dtype, _unstructure_dtype)


class StringEnum(StrEnum):
    @classmethod
    def has_value(cls, value: object, /) -> bool:
        return value in cls._value2member_map_

    @classmethod
    def to_tuple(cls) -> tuple[str, ...]:
        return tuple(member.value for member in cls)


def canonicalize_string_selection(
    str_values: str | Iterable[str],
    /,
    *,
    allowed: type[StringEnum],
    name: str,
) -> tuple[str, ...]:
    try:
        source = {str_values} if isinstance(str_values, str) else set(str_values)
    except TypeError as exc:
        raise TypeError(f"`{name}` must be a string or iterable of strings.") from exc

    normalized: list[str] = []
    for value in source:
        token = value.strip().lower()
        if not allowed.has_value(token):
            raise ValueError(
                f"Unknown {name} value {value!r}: Expected one of {allowed.to_tuple()}."
            )

        normalized.append(token)

    if len(normalized) == 0:
        raise ValueError(f"`{name}` must contain at least one value.")

    normalized_set = set(normalized)
    return tuple(value for value in allowed.to_tuple() if value in normalized_set)


def get_attrs_fields[T: attrs.AttrsInstance](cls: type[T], /) -> tuple[AttrsField, ...]:
    return cast(tuple[AttrsField, ...], attrs.fields(cls))


def get_attrs_fields_dict[T: attrs.AttrsInstance](cls: type[T], /) -> AttrsFields:
    return cast(AttrsFields, attrs.fields_dict(cls))


def insert_underscores(string: str, /) -> str:
    string = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", string)
    return re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", string)


def to_key_of_registry(string: str, /) -> str:
    return re.sub(r"[_ ]", "", string).lower()


def to_latex(instance: attrs.AttrsInstance, /, *, latex_name: str = "") -> str:
    latex_str = "$" + latex_name
    for label, attr in attrs.fields_dict(type(instance)).items():
        if attr.metadata.get("latex_name") is not None:
            latex_str += rf"\ {attr.metadata['latex_name']}={getattr(instance, label)}"
    return latex_str + "$"


def to_path(instance: attrs.AttrsInstance, /, *, root: Path) -> Path:
    for name, attr in attrs.fields_dict(type(instance)).items():
        dir_name = attr.metadata.get("dir_name")
        if isinstance(dir_name, str):
            value = str(cast(object, getattr(instance, name)))
            value = re.sub(r"[^\w\-.]", "_", value)
            root /= f"{dir_name}_{value.replace('.', 'p')}"
    return root


def canonicalize_source_dict(
    src: SourceDict,
    /,
    *,
    registry: dict[str, type[attrs.AttrsInstance]],
) -> dict[str, object]:
    if set(src) != {"type", "parameters"}:
        raise ValueError("Source keys must be exactly `type` and `parameters`.")

    type_name = src["type"]
    if not isinstance(type_name, str):
        raise TypeError("Configuration `type` must be a string.")

    parameters = src["parameters"]
    if not isinstance(parameters, dict):
        raise TypeError("Configuration `parameters` must be a dictionary.")

    registry_key = to_key_of_registry(type_name)
    try:
        registered_cls = registry[registry_key]
    except KeyError as exc:
        raise ValueError(f"Unknown configuration type {type_name!r}.") from exc

    fields = get_attrs_fields(registered_cls)
    init_field_names = {field.name for field in fields if field.init}
    parameter_names = set(parameters)

    if parameter_names != init_field_names:
        missing_init_fields = tuple(sorted(init_field_names - parameter_names))
        extra_parameters = tuple(sorted(parameter_names - init_field_names))
        raise ValueError(
            f"Invalid parameters for {registered_cls.__name__}: "
            + f" missing={missing_init_fields}, extra={extra_parameters}."
        )

    return {"type": registered_cls.__name__, "parameters": parameters}


def to_source_dict(instance: attrs.AttrsInstance, /) -> SourceDict:
    fields = get_attrs_fields(type(instance))
    return {
        "type": type(instance).__name__,
        "parameters": {
            field.name: to_json_compatible(cast(object, getattr(instance, field.name)))
            for field in fields
            if field.init
        },
    }


def to_json_compatible(value: object, /) -> object:
    if value is None or isinstance(value, bool | int | str):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite float {value!r} is not JSON-compatible.")
        return value

    if isinstance(value, complex):
        return {
            "real": to_json_compatible(value.real),
            "imag": to_json_compatible(value.imag),
        }

    if isinstance(value, bytes):
        return {"hex": value.hex()}

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, list | tuple):
        iterable = cast(list[object] | tuple[object, ...], value)
        return [to_json_compatible(item) for item in iterable]

    if isinstance(value, dict):
        mapping = cast(dict[object, object], value)
        for key in mapping:
            if not isinstance(key, str):
                raise TypeError(
                    f"JSON object keys must be strings, got {type(key).__name__}."
                )

        return {key: to_json_compatible(item) for key, item in mapping.items()}

    if isinstance(value, np.ndarray):
        return to_json_compatible(cast(list[object], value.tolist()))

    if isinstance(value, np.dtype):
        return value.name

    if isinstance(value, np.generic):
        return to_json_compatible(value.item())

    if isinstance(value, np.random.SeedSequence):
        return {
            "entropy": to_json_compatible(value.entropy),
            "spawn_key": list(value.spawn_key),
            "pool_size": value.pool_size,
        }

    if isinstance(value, np.random.Generator):
        return {
            "generator": type(value.bit_generator).__name__,
            "state": to_json_compatible(value.bit_generator.state),
        }

    if isinstance(value, np.random.BitGenerator):
        return {
            "bit_generator": type(value).__name__,
            "state": to_json_compatible(value.state),
        }

    if attrs.has(type(value)):
        return to_source_dict(value)

    raise TypeError(
        f"{type(value).__name__} cannot be converted to a JSON-compatible value."
    )


def build_hashed_id(array: NDArray[np.generic], /, *, num_hex: int = 16) -> str:
    hash_object = hashlib.sha256()
    hash_object.update(str(array.dtype).encode())
    hash_object.update(str(array.shape).encode())
    hash_object.update(array.tobytes())
    return hash_object.hexdigest()[:num_hex]
