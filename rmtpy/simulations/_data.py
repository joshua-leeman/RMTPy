from __future__ import annotations

import inspect
import os
import shutil
from abc import ABC
from pathlib import Path
from typing import Any

import numpy as np
from attrs import asdict, field, frozen
from numpy.lib.npyio import NpzFile

from ..utils import rmtpy_converter, insert_underscores


DATA_REGISTRY: dict[str, type[Data]] = {}


def load_data(path: str | Path) -> dict[str, Any]:
    return Data.load(path=Path(path))


def _normalize_metadata(metadata: dict | np.ndarray) -> dict[str, Any]:
    if isinstance(metadata, np.ndarray) and metadata.dtype == object:
        metadata = metadata.item()
    if isinstance(metadata, dict):
        return metadata
    raise TypeError(f"Expected dict, got {type(metadata).__name__}")


def _normalize_source(src: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(src, (str, Path)):
        with np.load(src, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}
    if isinstance(src, dict):
        return src
    raise TypeError(f"Expected path, dict, npz file, got {type(src).__name__}")


@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class Data(ABC):
    file_name: str = field(converter=lambda name: str(name) + "_data")
    metadata: dict[str, Any] = field(init=False, factory=dict, repr=False)

    @classmethod
    def __attrs_init_subclass__(cls) -> None:
        if not inspect.isabstract(cls):
            data_key: str = insert_underscores(cls.__name__)
            data_key = data_key.lower()
            DATA_REGISTRY[data_key] = cls

    @classmethod
    def load(cls, path: str | Path) -> Data:
        path: Path = Path(path)
        return rmtpy_converter.structure(path, cls)

    def __attrs_post_init__(self) -> None:
        key: str = insert_underscores(type(self).__name__)
        key = key.lower()
        self.metadata["name"] = key

    def save(self, path: str | Path) -> None:
        path: Path = Path(path)
        tmp_path: Path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "wb") as file:
            np.savez(file, **asdict(self), allow_pickle=True)
            file.flush()
            os.fsync(file.fileno())
        shutil.move(tmp_path, path)


@rmtpy_converter.register_structure_hook
def data_structure_hook(src: str | Path | dict[str, Any] | NpzFile | Data, _) -> Data:
    if isinstance(src, (str, Path)):
        file_name = Path(src).name

    src_dict: dict[str, Any] = _normalize_source(src)
    metadata: dict[str, Any] = _normalize_metadata(src_dict["metadata"])
    src_dict["metadata"] = metadata

    key: str | None = metadata.get("name", None)
    if key in DATA_REGISTRY:
        data_cls: type[Data] = DATA_REGISTRY[key]
    else:
        raise ValueError(f"No registered Data class found in {src}")

    data_instance: Data = data_cls(file_name=file_name)
    for key in src_dict:
        object.__setattr__(data_instance, key, src_dict[key])
    return data_instance
