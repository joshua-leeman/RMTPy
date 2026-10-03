import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TypeAliasType, cast, get_args, get_origin, get_type_hints

import attrs
import numpy as np

from ..conversion import AttrsFields, import_rmtpy_object, numpy_savez_serializer
from ..ensembles import RandomMatrixEnsemble

SAVED_CLASS_MODULE: str = "__module__"
SAVED_CLASS_QUALNAME: str = "__qualname__"


def load_saved_data(directory: str | Path) -> dict[str, Data]:
    loaded_data: dict[str, Data] = {}
    for data_path in sorted(Path(directory).rglob("*_data.npz")):
        data = Data.load(data_path)
        if data._file_name in loaded_data:
            raise ValueError(f"Saved data `{data._file_name}` is duplicated.")

        loaded_data[data._file_name] = data

    return loaded_data


def graft_loaded_data(value: object, loaded_data: dict[str, Data]) -> object:
    if isinstance(value, Data):
        try:
            replacement = loaded_data.pop(value._file_name)
        except KeyError as exc:
            raise ValueError(f"Saved data `{value._file_name}` is missing.") from exc
        if type(replacement) is not type(value):
            raise TypeError(
                f"Saved data `{value._file_name}` has class "
                + f"`{type(replacement).__name__}`."
            )

        return replacement

    if isinstance(value, RandomMatrixEnsemble):
        return value

    if isinstance(value, tuple):
        items = cast(tuple[object, ...], value)
        return tuple(graft_loaded_data(item, loaded_data) for item in items)

    if isinstance(value, list):
        items = cast(list[object], value)
        return [graft_loaded_data(item, loaded_data) for item in items]

    if attrs.has(type(value)):
        for field in cast(AttrsFields, attrs.fields(type(value))):
            current_attr = cast(object, getattr(value, field.name))
            updated_attr = graft_loaded_data(current_attr, loaded_data)
            if updated_attr is not current_attr:
                object.__setattr__(value, field.name, updated_attr)

        return value

    return value


def _decode_saved_value(
    array: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    annotation: object,
) -> object:
    if annotation is None:
        if array.ndim == 0:
            return cast(object, array.item())

        return array

    while isinstance(annotation, TypeAliasType):
        annotation = cast(object, annotation.__value__)

    origin = get_origin(annotation)
    if origin is dict:
        text = cast(object, array.item())
        if not isinstance(text, str):
            raise TypeError("Saved dictionary field is malformed.")

        mapping = cast(object, json.loads(text))
        if not isinstance(mapping, dict):
            raise TypeError("Saved dictionary field is malformed.")

        decoded = cast(dict[object, object], mapping)
        if not all(isinstance(key, str) for key in decoded):
            raise TypeError("Saved dictionary field is malformed.")

        return cast(dict[str, object], decoded)

    if origin is np.ndarray:
        return array

    if origin is tuple or origin is list:
        values = cast(object, array.tolist())
        if not isinstance(values, list):
            raise TypeError("Saved sequence field is malformed.")

        items = cast(list[object], values)
        if origin is tuple:
            return tuple(items)

        return items

    if array.ndim != 0:
        return array

    value = cast(object, array.item())
    if (
        isinstance(value, float)
        and math.isnan(value)
        and (annotation is type(None) or type(None) in get_args(annotation))
    ):
        return None

    return value


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class Data:
    metadata: dict[str, object] = attrs.field(
        factory=dict,
        converter=dict,
        repr=False,
    )

    _file_name: str = attrs.field(
        default="simulation",
        alias="_file_name",
        converter=str,
    )
    realizs: int = attrs.field(
        default=0,
        converter=int,
        validator=attrs.validators.ge(0),
        repr=False,
    )

    @property
    def to_path(self) -> Path:
        return Path(self._file_name) / Path(f"{self._file_name}_data.npz")

    @classmethod
    def load(cls, path: str | Path, /) -> Data:
        path = Path(path)

        if path.is_dir():
            path = path / f"{path.name}_data.npz"
        if path.suffix != ".npz" or not path.is_file():
            raise ValueError(f"Data archive `{path}` is malformed.")

        with cast(np.lib.npyio.NpzFile, np.load(path, allow_pickle=False)) as archive:
            if (
                SAVED_CLASS_MODULE not in archive.files
                or SAVED_CLASS_QUALNAME not in archive.files
            ):
                raise ValueError("Saved data class is missing.")

            module_name = cast(str, archive[SAVED_CLASS_MODULE].item())
            qualname = cast(str, archive[SAVED_CLASS_QUALNAME].item())

            loaded = import_rmtpy_object(qualname, module_name=module_name)
            if not isinstance(loaded, type) or not issubclass(loaded, Data):
                raise TypeError(f"Saved data `{qualname}` is malformed.")
            if cls is not Data and not issubclass(loaded, cls):
                raise TypeError(f"`{cls.__name__}` cannot load `{loaded.__name__}`.")

            annotations = get_type_hints(loaded)
            arguments: dict[str, object] = {}
            for field in cast(AttrsFields, attrs.fields(loaded)):
                if not field.init:
                    continue
                if field.name not in archive.files:
                    if field.metadata.get("archive_optional") is True:
                        continue
                    raise ValueError(f"Saved data is missing `{field.name}`.")

                saved_data = cast(
                    np.ndarray[tuple[int, ...], np.dtype[np.generic]],
                    archive[field.name],
                )
                annotation = cast(object, annotations.get(field.name, field.type))
                alias = field.name if field.alias is None else field.alias
                arguments[alias] = _decode_saved_value(saved_data, annotation)

            data_factory = cast(Callable[..., Data], loaded)
            return data_factory(**arguments)

    def attach_metadata(self, new_metadata: Mapping[str, object], /) -> None:
        self.metadata.update(new_metadata)

    def _aggregation_metadata(self) -> dict[str, object]:
        return self.metadata

    def _validate_contribution(self, contribution: Data, /) -> None:
        if type(contribution) is not type(self):
            raise TypeError(
                f"Cannot add `{type(contribution).__name__}` to "
                + f"`{type(self).__name__}`."
            )
        if contribution._file_name != self._file_name:
            raise ValueError(
                f"Data contribution `{contribution._file_name}` does not match "
                + f"`{self._file_name}`."
            )
        if contribution._aggregation_metadata() != self._aggregation_metadata():
            raise ValueError(
                f"Data contribution `{self._file_name}` has incompatible metadata."
            )

    def _add_realizations(self, contribution: Data, /) -> None:
        object.__setattr__(self, "realizs", self.realizs + contribution.realizs)

    def add_contribution(self, contribution: Data, /) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} has not implemented contribution aggregation."
        )

    def compute_statistics(self) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} has not implemented statistic finalization."
        )

    def save(self, *, directory: str | Path | None = None) -> None:
        path = self.to_path if directory is None else Path(directory) / self.to_path

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = cast(
            dict[str, np.ndarray[tuple[int, ...], np.dtype[np.generic]]],
            attrs.asdict(self, value_serializer=numpy_savez_serializer),
        )
        payload[SAVED_CLASS_MODULE] = np.array(type(self).__module__)
        payload[SAVED_CLASS_QUALNAME] = np.array(type(self).__qualname__)

        for name, array in payload.items():
            if array.dtype.hasobject:
                raise TypeError(f"Object array `{name}` is not supported.")

        np.savez_compressed(path, allow_pickle=False, **payload)
