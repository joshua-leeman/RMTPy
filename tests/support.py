import json
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import numpy as np

type ArchiveArray = np.ndarray[tuple[int, ...], np.dtype[np.generic]]
type FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.floating]]
type ComplexArray = np.ndarray[tuple[int, ...], np.dtype[np.complexfloating]]
type FloatVector = np.ndarray[tuple[int], np.dtype[np.floating]]
type NumericArray = np.ndarray[
    tuple[int, ...], np.dtype[np.integer | np.floating | np.complexfloating]
]


def archive_fields(path: Path, /) -> dict[str, ArchiveArray]:
    with cast(np.lib.npyio.NpzFile, np.load(path, allow_pickle=False)) as archive:
        return {name: cast(ArchiveArray, archive[name]) for name in archive.files}


def json_mapping(text: str, /) -> dict[str, object]:
    value = cast(object, json.loads(text))
    if not isinstance(value, dict):
        raise TypeError("Expected a JSON object.")

    return cast(dict[str, object], value)


def manifest_section(
    mapping: dict[str, object],
    *keys: str,
) -> dict[str, object]:
    for key in keys:
        value = mapping[key]
        if not isinstance(value, dict):
            raise TypeError(f"Expected a mapping at `{key}`.")

        mapping = cast(dict[str, object], value)

    return mapping


def mock_argument[T](mock: MagicMock, index: int, cls: type[T], /) -> T:
    arguments = mock.call_args
    if arguments is None:
        raise AssertionError("The mock has not been called.")

    value = cast(object, arguments.args[index])
    if not isinstance(value, cls):
        raise AssertionError(f"Expected a `{cls.__name__}` argument.")

    return value


def attribute_value(instance: object, name: str, /) -> object:
    return cast(object, getattr(instance, name))
