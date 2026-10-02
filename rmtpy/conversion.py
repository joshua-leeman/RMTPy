import base64
import hashlib
import importlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import cast

import attrs
import cattrs
import numpy as np
from numpy.typing import DTypeLike

type SourceDict = dict[str, str | dict[str, object]]
type AttrsFields = tuple[attrs.Attribute[object], ...]

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


def import_rmtpy_object(qualname: str, *, module_name: str) -> object:
    if not module_name.startswith("rmtpy."):
        raise ValueError(f"Module `{module_name}` is outside `rmtpy`.")

    imported: object = importlib.import_module(module_name)
    for token in qualname.split("."):
        if token == "":
            raise ValueError(f"Qualname `{qualname}` is malformed.")

        imported = cast(object, getattr(imported, token))

    return imported


def read_utc_time() -> str:
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
        if latex_label is None:
            continue
        elif isinstance(latex_label, str):
            latex_str += rf"\ {latex_label}={getattr(instance, label)}"
        else:
            raise ValueError("Attribute 'latex_label' must be a string.")

    return latex_str + "$"


def to_path(instance: attrs.AttrsInstance, /, *, root: Path) -> Path:
    for name, attr in attrs.fields_dict(type(instance)).items():
        dir_name = attr.metadata.get("dir_name")
        if dir_name is None:
            continue
        elif isinstance(dir_name, str):
            value = str(cast(object, getattr(instance, name)))
            value = re.sub(r"[^\w\-.]", "_", value)
            root /= f"{dir_name}_{value.replace('.', 'p')}"
        else:
            raise ValueError("Attribute 'dir_name' must be a string.")

    return root


def canonicalize_source_dict(
    source: SourceDict,
    /,
    *,
    registry: dict[str, type[attrs.AttrsInstance]],
) -> SourceDict:
    if set(source) != {"type", "parameters"}:
        raise ValueError("Source keys must be exactly `type` and `parameters`.")

    type_name = source["type"]
    if not isinstance(type_name, str):
        raise TypeError("Configuration `type` must be a string.")

    parameters = source["parameters"]
    if not isinstance(parameters, dict):
        raise TypeError("Configuration `parameters` must be a dictionary.")

    registry_key = to_key_of_registry(type_name)
    try:
        registered_cls = registry[registry_key]
    except KeyError as exc:
        raise ValueError(f"Unknown configuration type {type_name!r}.") from exc

    fields = cast(AttrsFields, attrs.fields(registered_cls))
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
    fields = cast(AttrsFields, attrs.fields(type(instance)))
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


def json_value_serializer(
    _instance: attrs.AttrsInstance,
    _field: attrs.Attribute[object],
    value: object,
) -> object:
    return json_value(value)


def unwrap_json_value(value: object) -> object:
    if isinstance(value, list):
        items = cast(list[object], value)
        return [unwrap_json_value(item) for item in items]

    if not isinstance(value, dict):
        return value

    mapping = cast(dict[str, object], value)
    type_tag = mapping.get(TYPE_KEY)

    if not isinstance(type_tag, str):
        return {key: unwrap_json_value(item) for key, item in mapping.items()}

    if type_tag == "float":
        float_value = mapping.get("value")
        if isinstance(float_value, float):
            return float_value
        if float_value == "inf":
            return float("inf")
        if float_value == "-inf":
            return float("-inf")

        return float("nan")

    if type_tag == "complex":
        real_part = cast(float, unwrap_json_value(mapping.get("real")))
        imag_part = cast(float, unwrap_json_value(mapping.get("imag")))
        return complex(real_part, imag_part)

    if type_tag == "bytes":
        bytes_value = mapping.get("base64")
        if not isinstance(bytes_value, str):
            raise TypeError("Tagged bytes value is malformed.")

        return base64.b64decode(bytes_value)

    if type_tag == "tuple":
        items = unwrap_json_value(mapping.get("items"))
        if not isinstance(items, list):
            raise TypeError("Tagged tuple is malformed.")

        return tuple(cast(list[object], items))

    if type_tag == "dtype":
        dtype_value = mapping.get("value")
        if not isinstance(dtype_value, str):
            raise TypeError("Tagged dtype is malformed.")

        return np.dtype(dtype_value)

    if type_tag == "ndarray":
        dtype_token = mapping.get("dtype")
        if not isinstance(dtype_token, str):
            raise TypeError("Tagged ndarray is malformed.")

        shape = cast(tuple[int, ...], mapping.get("shape"))
        array_values = unwrap_json_value(mapping.get("values"))

        array = np.asarray(array_values, dtype=np.dtype(dtype_token))
        return np.reshape(array, shape)

    if type_tag == "random.SeedSequence":
        entropy = unwrap_json_value(mapping.get("entropy"))
        if isinstance(entropy, bool):
            raise TypeError("Tagged SeedSequence entropy is boolean.")
        if not isinstance(entropy, int | np.ndarray | list | tuple | None):
            raise TypeError("Tagged SeedSequence entropy is malformed.")

        return np.random.SeedSequence(
            entropy=cast(Sequence[int] | None, entropy),
            spawn_key=cast(Sequence[int], mapping.get("spawn_key")),
            pool_size=cast(int, mapping.get("pool_size")),
        )

    if type_tag == "random.Generator":
        generator_name = mapping.get("generator")
        if not isinstance(generator_name, str):
            raise TypeError("Tagged bit generator is malformed.")

        candidate = cast(object, getattr(np.random, generator_name, None))
        if not isinstance(candidate, type):
            raise TypeError("Tagged bit generator is not a type.")
        if not issubclass(candidate, np.random.BitGenerator):
            raise TypeError("Tagged bit generator is malformed.")

        bit_generator = candidate()
        state = unwrap_json_value(mapping.get("state"))
        if not isinstance(state, dict):
            raise TypeError("Tagged bit generator state is malformed.")

        state_mapping = cast(dict[object, object], state)
        generator_state: dict[str, object] = {}
        for key, item in state_mapping.items():
            if not isinstance(key, str):
                raise TypeError("Tagged bit generator state must use string keys.")

            generator_state[key] = item

        bit_generator.state = generator_state
        return np.random.Generator(bit_generator)

    if type_tag == "random.BitGenerator":
        bit_generator_name = mapping.get("bit_generator")
        if not isinstance(bit_generator_name, str):
            raise TypeError("Tagged bit generator is malformed.")

        candidate = cast(object, getattr(np.random, bit_generator_name, None))
        if not isinstance(candidate, type):
            raise TypeError("Tagged bit generator is not a type.")
        if not issubclass(candidate, np.random.BitGenerator):
            raise TypeError("Tagged bit generator is malformed.")

        bit_generator = candidate()
        state = unwrap_json_value(mapping.get("state"))
        if not isinstance(state, dict):
            raise TypeError("Tagged bit generator state is malformed.")

        state_mapping = cast(dict[object, object], state)
        bit_generator_state: dict[str, object] = {}
        for key, item in state_mapping.items():
            if not isinstance(key, str):
                raise TypeError("Tagged bit generator state must use string keys.")

            bit_generator_state[key] = item

        bit_generator.state = bit_generator_state
        return bit_generator

    raise ValueError(f"Unknown manifest tag `{type_tag}`.")


def numpy_savez_value(
    value: object, /
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    if isinstance(value, dict):
        value = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
        )

    if value is None:
        return np.array(np.nan, dtype=np.float64)

    if isinstance(value, bool | str):
        return np.array(value)

    if isinstance(value, int):
        return np.array(value, dtype=np.int64)

    if isinstance(value, float):
        return np.array(value, dtype=np.float64)

    if isinstance(value, complex):
        return np.array(value, dtype=np.complex128)

    if isinstance(value, list | tuple):
        value = [numpy_savez_value(val) for val in cast(Sequence[object], value)]
        return np.asarray(value)

    if isinstance(value, np.ndarray):
        return value

    raise TypeError(f"Unsupported value: {type(value).__name__}.")


def numpy_savez_serializer(
    _instance: attrs.AttrsInstance,
    _field: attrs.Attribute[object],
    value: object,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    return numpy_savez_value(value)
