from __future__ import annotations

import inspect
import json
import re
from abc import ABC
from pathlib import Path
from typing import Any

from attrs import asdict, field, fields, fields_dict, frozen
from attrs.validators import gt
from cattrs.dispatch import StructureHook, UnstructureHook

from ._data import DATA_REGISTRY, Data
from ._plot import Plot
from ..ensembles import RandomMatrixEnsemble
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
    ensemble: RandomMatrixEnsemble = field(converter=RandomMatrixEnsemble.create)

    @ensemble.validator
    def _ensemble_is_concrete(self, _, value: RandomMatrixEnsemble) -> None:
        if inspect.isabstract(value):
            raise ValueError(f"Ensemble must be concrete.")

    realizs: int = field(
        converter=int,
        validator=gt(0),
        metadata={"dir_name": "realizs", "latex_name": "R"},
    )

    metadata: dict[str, Any] = field(init=False, factory=dict, repr=False)
    dir_name: str = field(init=False, repr=False)

    @dir_name.default
    def _dir_name_default(self) -> str:
        return insert_underscores(type(self).__name__).lower()

    @classmethod
    def __attrs_init_subclass__(cls) -> None:
        if not inspect.isabstract(cls):
            sim_key: str = re.sub(r"_", "", cls.__name__).lower()
            SIMULATION_REGISTRY[sim_key] = cls

    def __attrs_post_init__(self) -> None:
        self.metadata["name"] = type(self).__name__
        self.metadata["args"] = {}
        self.metadata["args"]["ensemble"] = rmtpy_converter.unstructure(self.ensemble)
        self.metadata["args"]["realizs"] = self.realizs
        data_attrs: tuple[Data, ...] = tuple(
            getattr(self, f.name)
            for f in fields(type(self))
            if isinstance(getattr(self, f.name), Data)
        )
        for data in data_attrs:
            data.metadata.update({"simulation": self.metadata.copy()})

    def save_data(self, out_dir: str | Path) -> None:
        with open(out_dir / "metadata.json", "w") as file:
            json.dump(self.metadata, file, indent=4, default=str)

        data_attrs: tuple[Data, ...] = tuple(
            getattr(self, f.name)
            for f in fields(type(self))
            if isinstance(getattr(self, f.name), Data)
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
            getattr(self, f.name)
            for f in fields(type(self))
            if isinstance(getattr(self, f.name), Plot)
        )
        for plot in plot_attrs:
            subdir_name = plot.data.file_name.replace("_data", "")
            out_path: Path = out_dir / subdir_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plot.plot(path=out_path)

    @property
    def to_dir(self) -> Path:
        self_asdict: dict[str, Any] = asdict(self)
        dir_path: Path = Path(self._dir_name)
        dir_path /= self.ensemble.to_dir
        for name, attr in fields_dict(type(self)).items():
            if attr.metadata.get("dir_name", None) is not None:
                val: str = re.sub(r"[^\w\-.]", "_", str(self_asdict[name]))
                dir_path /= f"{attr.metadata['dir_name']}_{val.replace('.', 'p')}"
        return dir_path

    @property
    def _dir_name(self) -> str:
        return insert_underscores(type(self).__name__).lower()


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

    sim_dict: dict[str, Any] = normalize_dict(metadata, SIMULATION_REGISTRY)
    sim_name: str = sim_dict.pop("name")
    if not isinstance(sim_name, str):
        raise ValueError(f"Invalid simulation name type: {type(sim_name).__name__}")

    sim_key: str = re.sub(r"_", "", sim_name).lower()
    sim_cls: type[Simulation] = SIMULATION_REGISTRY[sim_key]
    sim_args: dict[str, Any] = sim_dict.pop("args")
    if not isinstance(sim_args, dict):
        raise ValueError(f"Invalid simulation args type: {type(sim_args).__name__}")

    sim_inst: Simulation = SIMULATION_STRUCTURE_HOOKS[sim_key](sim_args, sim_cls)
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
