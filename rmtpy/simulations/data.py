from __future__ import annotations

import inspect
import os
import shutil
from pathlib import Path
from typing import Any

import attrs
import numpy as np
from numpy.lib.npyio import NpzFile

import rmtpy.conversion
from rmtpy.conversion import RMT_CONVERTER

REGISTRY: dict[str, type[Data]] = {}


def load_data(path: str | Path) -> dict[str, Any]:
    return Data.load(path=Path(path))


def normalize_metadata(metadata: dict | np.ndarray) -> dict[str, Any]:
    if isinstance(metadata, np.ndarray) and metadata.dtype == object:
        metadata = metadata.item()
    if isinstance(metadata, dict):
        return metadata
    raise TypeError(f"Expected dict, got {type(metadata).__name__}")


def normalize_source(src: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(src, (str, Path)):
        with np.load(src, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}
    if isinstance(src, dict):
        return src
    raise TypeError(f"Expected path, dict, npz file, got {type(src).__name__}")


def normalize_saved_value(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def file_name_for_init(value: Any) -> str:
    file_name = str(normalize_saved_value(value))
    if file_name.endswith(".npz"):
        file_name = Path(file_name).stem
    if file_name.endswith("_data"):
        file_name = file_name[: -len("_data")]
    return file_name


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class Data:
    file_name: str = attrs.field(
        default="simulation",
        converter=lambda name: str(name) + "_data",
    )
    metadata: dict[str, Any] = attrs.field(
        factory=dict,
        init=False,
        repr=False,
    )

    def __attrs_post_init__(self) -> None:
        key: str = rmtpy.conversion.insert_underscores(type(self).__name__)
        key = key.lower()
        self.metadata["name"] = key

    @classmethod
    def __attrs_init_subclass__(cls) -> None:
        if not inspect.isabstract(cls):
            data_key: str = rmtpy.conversion.insert_underscores(cls.__name__)
            data_key = data_key.lower()
            REGISTRY[data_key] = cls

    @classmethod
    def load(cls, path: str | Path) -> Data:
        path: Path = Path(path)
        return RMT_CONVERTER.structure(path, cls)

    def save(self, path: str | Path) -> None:
        path: Path = Path(path)
        tmp_path: Path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "wb") as file:
            np.savez(file, **attrs.asdict(self), allow_pickle=True)
            file.flush()
            os.fsync(file.fileno())
        shutil.move(tmp_path, path)


@RMT_CONVERTER.register_structure_hook
def data_structure_hook(src: str | Path | dict[str, Any] | NpzFile | Data, _) -> Data:
    if isinstance(src, (str, Path)):
        file_name = Path(src).name
    else:
        file_name = src.get("file_name", "data") if isinstance(src, dict) else "data"

    src_dict: dict[str, Any] = normalize_source(src)
    metadata: dict[str, Any] = normalize_metadata(src_dict["metadata"])
    src_dict["metadata"] = metadata

    key: str | None = metadata.get("name")
    if key in REGISTRY:
        data_cls: type[Data] = REGISTRY[key]
    else:
        raise ValueError(f"No registered Data class found in {src}")

    init_kwargs: dict[str, Any] = {}
    for name, attr in attrs.fields_dict(data_cls).items():
        if not attr.init:
            continue
        if name == "file_name":
            init_kwargs[name] = file_name_for_init(src_dict.get(name, file_name))
        elif name in src_dict:
            init_kwargs[name] = normalize_saved_value(src_dict[name])

    data_instance: Data = data_cls(**init_kwargs)
    for key, value in src_dict.items():
        object.__setattr__(data_instance, key, normalize_saved_value(value))
    return data_instance
