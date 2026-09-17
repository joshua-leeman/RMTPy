import hashlib
import re
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import cast

import attrs
import cattrs
import numpy as np
from numpy.typing import DTypeLike, NDArray


def structure_dtype(value: DTypeLike, _: object) -> np.dtype[np.generic]:
    return np.dtype(value)


def unstructure_dtype(dtype: np.dtype[np.generic]) -> str:
    return dtype.name


RMT_CONVERTER: cattrs.Converter = cattrs.Converter()
RMT_CONVERTER.register_structure_hook(np.dtype, structure_dtype)
RMT_CONVERTER.register_unstructure_hook(np.dtype, unstructure_dtype)


class StringEnum(StrEnum):
    @classmethod
    def has_value(cls, value: object) -> bool:
        return value in cls._value2member_map_

    @classmethod
    def to_tuple(cls) -> tuple[str, ...]:
        return tuple(member.value for member in cls)


def to_latex(instance: attrs.AttrsInstance, *, latex_name: str = "") -> str:
    latex_str = "$" + latex_name
    for label, attr in attrs.fields_dict(type(instance)).items():
        if attr.metadata.get("latex_name") is not None:
            latex_str += rf"\ {attr.metadata['latex_name']}={getattr(instance, label)}"
    return latex_str + "$"


def to_path(instance: attrs.AttrsInstance, *, root: Path) -> Path:
    for name, attr in attrs.fields_dict(type(instance)).items():
        dir_name = attr.metadata.get("dir_name")
        if isinstance(dir_name, str):
            value = str(cast(object, getattr(instance, name)))
            value = re.sub(r"[^\w\-.]", "_", value)
            root /= f"{dir_name}_{value.replace('.', 'p')}"
    return root


def to_registry_key(string: str) -> str:
    return re.sub(r"[_ ]", "", string).lower()


def insert_underscores(string: str) -> str:
    string = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", string)
    return re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", string)


def build_hashed_id(array: NDArray[np.generic], *, num_hex: int = 16) -> str:
    hash_object = hashlib.sha256()
    hash_object.update(str(array.dtype).encode())
    hash_object.update(str(array.shape).encode())
    hash_object.update(array.tobytes())
    return hash_object.hexdigest()[:num_hex]


def canonicalize_string_selection(
    values: str | Iterable[str],
    *,
    allowed: type[StringEnum],
    name: str,
) -> tuple[str, ...]:
    try:
        source = {values} if isinstance(values, str) else set(values)
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


def normalize_source(
    src: dict[str, str | dict[str, object]],
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

    registry_key = to_registry_key(type_name)
    try:
        registered_cls = registry[registry_key]
    except KeyError as exc:
        raise ValueError(f"Unknown configuration type {type_name!r}.") from exc

    fields = cast(tuple[attrs.Attribute[object], ...], attrs.fields(registered_cls))
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


def normalize_value(value: object) -> object:
    if value is None or isinstance(value, bool | int | float | str):
        return value

    if isinstance(value, bytes):
        return {"hex": value.hex()}

    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}

    if isinstance(value, list | tuple):
        iterable = cast(list[object] | tuple[object, ...], value)
        return [normalize_value(item) for item in iterable]

    if isinstance(value, dict):
        mapping = cast(dict[object, object], value)
        if any(not isinstance(key, str) for key in mapping):
            raise TypeError("Configuration mappings must use string keys.")

        return {key: normalize_value(item) for key, item in mapping.items()}

    if isinstance(value, np.ndarray):
        return normalize_value(cast(list[object], value.tolist()))

    if isinstance(value, np.dtype):
        return value.name

    if isinstance(value, np.generic):
        return normalize_value(value.item())

    if isinstance(value, np.random.SeedSequence):
        return {
            "entropy": normalize_value(value.entropy),
            "spawn_key": list(value.spawn_key),
            "pool_size": value.pool_size,
        }

    if isinstance(value, np.random.Generator):
        return {
            "generator": type(value.bit_generator).__name__,
            "state": normalize_value(value.bit_generator.state),
        }

    if isinstance(value, np.random.BitGenerator):
        return {
            "bit_generator": type(value).__name__,
            "state": normalize_value(value.state),
        }

    if attrs.has(type(value)):
        fields = cast(tuple[attrs.Attribute[object], ...], attrs.fields(type(value)))
        return {
            "type": type(value).__name__,
            "parameters": {
                field.name: normalize_value(cast(object, getattr(value, field.name)))
                for field in fields
                if field.init
            },
        }

    raise TypeError(f"Configuration cannot contain {type(value).__name__} values.")
