from __future__ import annotations

import inspect
import json
import re
from abc import ABC
from pathlib import Path
from typing import Any

from attrs import asdict, field, fields, fields_dict, frozen
from cattrs.dispatch import StructureHook, UnstructureHook

from ._data import DATA_REGISTRY, Data
from ._plot import Plot
from ..utils import rmtpy_converter, insert_underscores, normalize_dict


SIMULATION_REGISTRY: dict[str, type[Simulation]] = {}
SIMULATION_STRUCTURE_HOOKS: dict[str, StructureHook] = {
    key: rmtpy_converter.get_structure_hook(val)
    for key, val in SIMULATION_REGISTRY.items()
}
SIMULATION_UNSTRUCTURE_HOOKS: dict[str, UnstructureHook] = {
    key: rmtpy_converter.get_unstructure_hook(val)
    for key, val in SIMULATION_REGISTRY.items()
}


@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class Simulation(ABC):
    metadata: dict[str, Any] = field(init=False, factory=dict, repr=False)

    def __attrs_post_init__(self) -> None:
        self._populate_metadata()

        data_attrs: tuple[Data, ...] = tuple(
            getattr(self, simulation_field.name)
            for simulation_field in fields(type(self))
            if isinstance(getattr(self, simulation_field.name), Data)
        )
        for data in data_attrs:
            data.metadata.update({"simulation": self.metadata.copy()})

    @classmethod
    def __attrs_init_subclass__(cls) -> None:
        if not inspect.isabstract(cls):
            sim_key: str = re.sub(r"_", "", cls.__name__).lower()
            SIMULATION_REGISTRY[sim_key] = cls

    def _populate_metadata(self) -> None:
        self.metadata["name"] = type(self).__name__
        self.metadata["args"] = {}

    def _save_metadata(self, out_dir: str | Path) -> None:
        with open(out_dir / "metadata.json", "w") as file:
            json.dump(self.metadata, file, indent=4, default=str)

    def save_data(self, out_dir: str | Path) -> None:
        self._save_metadata(out_dir)

        data_attrs: tuple[Data, ...] = tuple(
            getattr(self, simulation_field.name)
            for simulation_field in fields(type(self))
            if isinstance(getattr(self, simulation_field.name), Data)
        )
        for data in data_attrs:
            subdir_name = data.file_name.replace("_data", "")
            out_path: Path = out_dir / subdir_name / f"{data.file_name}.npz"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            data.save(out_path)

    def initialize_plots(self) -> None:
        pass

    def save_plots(self, out_dir: str | Path) -> None:
        self.initialize_plots()
        plot_attrs: tuple[Plot, ...] = tuple(
            getattr(self, simulation_field.name)
            for simulation_field in fields(type(self))
            if isinstance(getattr(self, simulation_field.name), Plot)
        )
        for plot in plot_attrs:
            subdir_name = plot.data.file_name.replace("_data", "")
            out_path: Path = out_dir / subdir_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plot.plot(path=out_path)

    @property
    def _path_name(self) -> str:
        return insert_underscores(type(self).__name__).lower()

    @property
    def to_path(self) -> Path:
        self_asdict: dict[str, Any] = asdict(self)
        path: Path = Path(self._path_name)
        for name, attr in fields_dict(type(self)).items():
            if attr.metadata.get("dir_name", None) is not None:
                val: str = re.sub(r"[^\w\-.]", "_", str(self_asdict[name]))
                path /= f"{attr.metadata['dir_name']}_{val.replace('.', 'p')}"
        return path


@rmtpy_converter.register_structure_hook
def simulation_structure_hook(
    src: str | Path | dict[str, Any] | Simulation, _
) -> Simulation:
    if type(src) in SIMULATION_REGISTRY.values():
        return src
    elif isinstance(src, (str, Path)):
        path: Path = Path(src)
        with open(path / "metadata.json", "r") as file:
            metadata: dict[str, Any] = json.load(file)
    elif isinstance(src, dict):
        metadata: dict[str, Any] = src
    else:
        raise TypeError(f"Expected str, Path, dict, got {type(src).__name__}")

    simulation_dict: dict[str, Any] = normalize_dict(metadata, SIMULATION_REGISTRY)
    simulation_name: str = simulation_dict.pop("name")
    if not isinstance(simulation_name, str):
        raise ValueError(
            f"Invalid simulation name type: {type(simulation_name).__name__}"
        )

    key: str = re.sub(r"_", "", simulation_name).lower()
    simulation_cls: type[Simulation] = SIMULATION_REGISTRY[key]
    simulation_args: dict[str, Any] = simulation_dict.pop("args")
    if not isinstance(simulation_args, dict):
        raise ValueError(
            f"Invalid simulation args type: {type(simulation_args).__name__}"
        )

    simulation_inst: Simulation = SIMULATION_STRUCTURE_HOOKS[key](
        simulation_args, simulation_cls
    )
    if isinstance(src, (str, Path)):
        data_dirs: tuple[Path, ...] = tuple(
            folder for folder in path.iterdir() if folder.is_dir()
        )
        for folder in data_dirs:
            data_cls: type[Data] | None = DATA_REGISTRY.get(folder.name, None)
            if data_cls is None:
                continue
            data: Data = data_cls.load(folder / f"{folder.name}.npz")
            object.__setattr__(simulation_inst, folder.name + "_data", data)
    return simulation_inst
