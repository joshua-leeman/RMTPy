import hashlib
import re
from pathlib import Path
from typing import Any

import attrs
import cattrs
import numpy as np

RMTPY_CONVERTER: cattrs.Converter = cattrs.Converter()
RMTPY_CONVERTER.register_unstructure_hook(np.dtype, lambda dtype: np.dtype(dtype).name)
RMTPY_CONVERTER.register_structure_hook(np.dtype, lambda dtype, _: np.dtype(dtype))


def create_hashed_id(array: np.ndarray, num_hex: int = 16) -> str:
    hash_object: hashlib._Hash = hashlib.sha256()
    hash_object.update(str(array.dtype).encode())
    hash_object.update(str(array.shape).encode())
    hash_object.update(array.tobytes())
    return hash_object.hexdigest()[:num_hex]


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

    cls_attrs: dict[str, attrs.Attribute] = attrs.fields_dict(registered_cls)
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


def to_latex(instance: attrs.AttrsInstance, latex_name: str = "") -> str:
    latex_str: str = "$" + latex_name
    for label, attr in attrs.fields_dict(instance).items():
        if attr.metadata.get("latex_name") is not None:
            latex_str += rf"\ {attr.metadata['latex_name']}={getattr(instance, label)}"
    return latex_str + "$"


def to_path(instance: attrs.AttrsInstance, root: Path) -> Path:
    for name, attr in attrs.fields_dict(instance).items():
        if attr.metadata.get("dir_name") is not None:
            value: str = re.sub(r"[^\w\-.]", "_", str(getattr(instance, name)))
            root /= f"{attr.metadata['dir_name']}_{value.replace('.', 'p')}"
    return root


def to_registry_key(string: str) -> str:
    return re.sub(r"[_ ]", "", string).lower()
