import base64
import hashlib
import math
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import cast

import attrs
import cattrs
import numpy as np
from numpy.typing import DTypeLike

type SourceDict = dict[str, str | dict[str, object]]
type AttrsField = attrs.Attribute[object]
type AttrsFields = dict[str, AttrsField]

TYPE_KEY: str = "__rmtpy_manifest_type__"

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


def completed_at_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def build_hashed_id(
    array: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    /,
    *,
    num_hex: int = 16,
) -> str:
    hash_object = hashlib.sha256()
    hash_object.update(str(array.dtype).encode())
    hash_object.update(str(array.shape).encode())
    hash_object.update(array.tobytes())
    return hash_object.hexdigest()[:num_hex]


def insert_underscores(string: str, /) -> str:
    string = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", string)
    return re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", string)


def to_key_of_registry(string: str, /) -> str:
    return re.sub(r"[_ ]", "", string).lower()


def to_latex(instance: attrs.AttrsInstance, /, *, latex_name: str = "") -> str:
    latex_str = "$" + latex_name
    for label, attr in attrs.fields_dict(type(instance)).items():
        latex_label = attr.metadata.get("latex_name")
        if isinstance(latex_label, str):
            latex_str += rf"\ {latex_label}={getattr(instance, label)}"

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

    fields = cast(tuple[AttrsField, ...], attrs.fields(registered_cls))
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


def source_dict(instance: attrs.AttrsInstance, /) -> SourceDict:
    fields = cast(tuple[AttrsField, ...], attrs.fields(type(instance)))
    return {
        "type": type(instance).__name__,
        "parameters": {
            field.name: json_value(cast(object, getattr(instance, field.name)))
            for field in fields
            if field.init
        },
    }


def json_value(value: object, /) -> object:
    if value is None or isinstance(value, bool | int | str):
        return value

    if isinstance(value, Path):
        return str(value)

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
            "real": json_value(value.real),
            "imag": json_value(value.imag),
        }

    if isinstance(value, bytes):
        return {
            TYPE_KEY: "bytes",
            "base64": base64.b64encode(value).decode("ascii"),
        }

    if isinstance(value, tuple):
        tuple_items = cast(tuple[object, ...], value)
        return {TYPE_KEY: "tuple", "items": [json_value(item) for item in tuple_items]}

    if isinstance(value, list):
        list_items = cast(list[object], value)
        return [json_value(item) for item in list_items]

    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        normalized: dict[str, object] = {}
        for key, item in mapping.items():
            if not isinstance(key, str):
                raise TypeError("Mappings must use string keys.")
            if key == TYPE_KEY:
                raise ValueError(f"{TYPE_KEY!r} is reserved.")

            normalized[key] = json_value(item)

        return normalized

    if isinstance(value, np.dtype):
        return {TYPE_KEY: "dtype", "value": value.str}

    if isinstance(value, np.generic):
        return json_value(value.item())

    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object arrays are not supported.")

        return {
            TYPE_KEY: "ndarray",
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "values": json_value(cast(list[object], value.tolist())),
        }

    if isinstance(value, np.random.SeedSequence):
        return {
            TYPE_KEY: "random.SeedSequence",
            "entropy": json_value(value.entropy),
            "spawn_key": list(value.spawn_key),
            "pool_size": value.pool_size,
        }

    if isinstance(value, np.random.Generator):
        return {
            TYPE_KEY: "random.Generator",
            "generator": type(value.bit_generator).__name__,
            "state": json_value(value.bit_generator.state),
        }

    if isinstance(value, np.random.BitGenerator):
        return {
            TYPE_KEY: "random.BitGenerator",
            "bit_generator": type(value).__name__,
            "state": json_value(value.state),
        }

    if attrs.has(type(value)):
        return source_dict(value)

    raise TypeError(f"Unsupported value: {type(value).__name__}.")
