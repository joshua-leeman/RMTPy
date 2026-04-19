import re
from typing import Any

import numpy as np
from attrs import Attribute, fields_dict
from cattrs import Converter

rmtpy_converter = Converter()
rmtpy_converter.register_unstructure_hook(np.dtype, lambda dtype: dtype.name)
rmtpy_converter.register_structure_hook(np.dtype, lambda s, _: np.dtype(s))


def insert_underscores(string: str) -> str:
    string = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", string)
    return re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", string)


def normalize_dict(src: dict[str, Any], registry: dict[str, type]) -> dict[str, Any]:
    if not isinstance(src, dict):
        raise TypeError(f"Expected a dictionary, got {type(src).__name__}.")

    normalized_dict: dict[str, Any] = {}
    for val in src.values():
        if not isinstance(val, str):
            continue
        key: str = to_registry_key(val)
        if key in registry:
            registered_cls: type = registry[key]
            normalized_dict["name"] = registered_cls.__name__
            break

    if normalized_dict.get("name") is None:
        raise KeyError("Registered class name not found in dictionary as value.")

    cls_attrs: dict[str, Attribute] = fields_dict(registered_cls)
    cls_args: set[str] = {arg for arg, attr in cls_attrs.items() if attr.init}
    arg_dict: dict[str, Any] = {}

    for val in src.values():
        if not isinstance(val, dict):
            continue
        for key in val.keys():
            if key in cls_args:
                arg_dict[key] = val[key]

    if arg_dict == {}:
        arg_dict = {arg: src[arg] for arg in src if arg in cls_args}

    normalized_dict.update({"args": arg_dict})
    return normalized_dict


def to_registry_key(string: str) -> str:
    return re.sub(r"[_ ]", "", string).lower()
