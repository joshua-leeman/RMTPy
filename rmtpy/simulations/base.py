from __future__ import annotations

import inspect
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import attrs
from cattrs.dispatch import StructureHook, UnstructureHook

import rmtpy.conversion
from rmtpy.conversion import RMT_CONVERTER

from .data import REGISTRY as DATA_REGISTRY
from .data import Data
from .observable import Observable

REGISTRY: dict[str, type[Simulation]] = {}
STRUCTURE_HOOKS: dict[str, StructureHook] = {
    key: RMT_CONVERTER.get_structure_hook(val) for key, val in REGISTRY.items()
}
UNSTRUCTURE_HOOKS: dict[str, UnstructureHook] = {
    key: RMT_CONVERTER.get_unstructure_hook(val) for key, val in REGISTRY.items()
}


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class Simulation:
    metadata: dict[str, Any] = attrs.field(
        factory=dict,
        init=False,
        repr=False,
    )

    def __attrs_post_init__(self) -> None:
        self.populate_metadata()

        for observable in self.iter_observables():
            observable.metadata.update({"simulation": self.metadata.copy()})

    @classmethod
    def __attrs_init_subclass__(cls) -> None:
        if not inspect.isabstract(cls):
            sim_key: str = re.sub(r"_", "", cls.__name__).lower()
            REGISTRY[sim_key] = cls
            STRUCTURE_HOOKS[sim_key] = RMT_CONVERTER.get_structure_hook(cls)
            UNSTRUCTURE_HOOKS[sim_key] = RMT_CONVERTER.get_unstructure_hook(cls)

    @property
    def path_name(self) -> str:
        return rmtpy.conversion.insert_underscores(type(self).__name__).lower()

    @property
    def to_path(self) -> Path:
        path: Path = Path(self.path_name)
        for name, attr in attrs.fields_dict(type(self)).items():
            if attr.metadata.get("dir_name") is not None:
                val: str = re.sub(r"[^\w\-.]", "_", str(getattr(self, name)))
                path /= f"{attr.metadata['dir_name']}_{val.replace('.', 'p')}"
        return path

    def populate_metadata(self) -> None:
        self.metadata["name"] = type(self).__name__
        self.metadata["args"] = {}

    def iter_attrs_of_type(self, cls: type) -> Iterator[object]:
        for attr in attrs.fields(type(self)):
            value = getattr(self, attr.name)
            if isinstance(value, cls):
                yield value
            elif isinstance(value, list | tuple):
                yield from (item for item in value if isinstance(item, cls))

    def iter_observables(self) -> Iterator[Observable]:
        return self.iter_attrs_of_type(Observable)

    def realize_monte_carlo_simulation(self) -> None:
        pass

    def calculate_statistics(self) -> None:
        for observable in self.iter_observables():
            observable.calculate_statistics()

    def save_metadata(self, out_dir: str | Path) -> None:
        with open(out_dir / "metadata.json", "w") as file:
            json.dump(self.metadata, file, indent=4, default=str)

    def save_data(self, out_dir: str | Path) -> None:
        self.save_metadata(out_dir)

        for observable in self.iter_observables():
            observable.save_data(Path(out_dir))

    def save_plots(self, out_dir: str | Path) -> None:
        for observable in self.iter_observables():
            observable.initialize_plot()
            observable.save_plot(Path(out_dir))

    def run(self, out_dir: str | Path = "output") -> None:
        self.realize_monte_carlo_simulation()
        self.calculate_statistics()

        out_dir: Path = Path(out_dir)
        base_dir: Path = out_dir / self.to_path
        base_dir.mkdir(parents=True, exist_ok=True)

        self.save_data(out_dir=base_dir)
        self.save_plots(out_dir=base_dir)


@RMT_CONVERTER.register_structure_hook
def structure_hook_for_simulation(
    src: str | Path | dict[str, Any] | Simulation, _
) -> Simulation:
    if type(src) in REGISTRY.values():
        return src
    elif isinstance(src, (str, Path)):
        path: Path = Path(src)
        with open(path / "metadata.json") as file:
            metadata: dict[str, Any] = json.load(file)
    elif isinstance(src, dict):
        metadata: dict[str, Any] = src
    else:
        raise TypeError(f"Expected str, Path, dict, got {type(src).__name__}")

    sim_dict: dict[str, Any] = rmtpy.conversion.normalize_dict(metadata, REGISTRY)
    sim_name: str = sim_dict.pop("name")
    if not isinstance(sim_name, str):
        raise ValueError(f"Invalid simulation name type: {type(sim_name).__name__}")

    key: str = re.sub(r"_", "", sim_name).lower()
    sim_cls: type[Simulation] = REGISTRY[key]
    sim_args: dict[str, Any] = sim_dict.pop("args")
    if not isinstance(sim_args, dict):
        raise ValueError(f"Invalid simulation args type: {type(sim_args).__name__}")

    sim_inst: Simulation = STRUCTURE_HOOKS[key](sim_args, sim_cls)
    if isinstance(src, (str, Path)):
        data_dirs: tuple[Path, ...] = tuple(
            folder for folder in path.iterdir() if folder.is_dir()
        )
        for folder in data_dirs:
            data_cls: type[Data] | None = DATA_REGISTRY.get(folder.name, None)
            if data_cls is None:
                continue
            data: Data = data_cls.load(folder / f"{folder.name}.npz")
            object.__setattr__(sim_inst, folder.name + "_data", data)
    return sim_inst
