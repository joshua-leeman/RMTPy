from __future__ import annotations

import inspect
import re
from abc import ABC
from ast import literal_eval
from pathlib import Path
from typing import Any, Sequence, Union

import numpy as np
from attrs import asdict, field, fields_dict, frozen
from attrs.validators import gt
from cattrs.dispatch import StructureHook, UnstructureHook

from ..utils import rmtpy_converter, insert_underscores, normalize_dict, to_registry_key

SeedLike = Union[
    None,
    int,
    Sequence[int],
    np.random.SeedSequence,
    np.random.BitGenerator,
    np.random.Generator,
]

ENSEMBLE_REGISTRY: dict[str, type[RandomMatrixEnsemble]] = {}
ENSEMBLE_STRUCTURE_HOOKS: dict[str, StructureHook] = {}
ENSEMBLE_UNSTRUCTURE_HOOKS: dict[str, UnstructureHook] = {}


def create_random_matrix_ensemble(**kwargs: Any) -> RandomMatrixEnsemble:
    return RandomMatrixEnsemble.create(kwargs)


@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class RandomMatrixEnsemble(ABC):
    dimension: int = field(
        converter=int,
        validator=gt(0),
        metadata={"dir_name": "dim", "latex_name": "D"},
    )
    dtype: np.dtype = field(default=np.dtype("complex128"), converter=np.dtype)
    seed: SeedLike = field(
        default=None,
        converter=lambda s: literal_eval(s) if isinstance(s, str) else s,
    )

    complex_dtype: np.dtype = field(init=False, repr=False)

    @complex_dtype.default
    def _default_complex_dtype(self) -> np.dtype:
        return np.dtype(self.dtype.char.upper())

    real_dtype: np.dtype = field(init=False, repr=False)

    @real_dtype.default
    def _default_real_dtype(self) -> np.dtype:
        return np.dtype(self.dtype.char.lower())

    rng: np.random.Generator = field(init=False, repr=False)

    @rng.default
    def _default_rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

    _nickname: str = field(init=False, default="RME", repr=False)

    @classmethod
    def __attrs_init_subclass__(cls) -> None:
        if not inspect.isabstract(cls):
            key: str = to_registry_key(cls.__name__)
            ENSEMBLE_REGISTRY[key] = cls
            ENSEMBLE_STRUCTURE_HOOKS[key] = rmtpy_converter.get_structure_hook(cls)
            ENSEMBLE_UNSTRUCTURE_HOOKS[key] = rmtpy_converter.get_unstructure_hook(cls)

    @classmethod
    def create(cls, src: dict[str, Any] | RandomMatrixEnsemble) -> RandomMatrixEnsemble:
        return rmtpy_converter.structure(src, cls)

    @property
    def _path_name(self) -> str:
        return insert_underscores(self._nickname)

    @property
    def _latex_name(self) -> str:
        return f"\\textrm{{{re.sub(r'_', ' ', self._nickname)}}}"

    @property
    def to_path(self) -> Path:
        path: Path = Path(self._path_name)
        self_asdict: dict[str, Any] = asdict(self)
        for name, attr in fields_dict(type(self)).items():
            if attr.metadata.get("dir_name") is not None:
                val: str = re.sub(r"[^\w\-.]", "_", str(self_asdict[name]))
                path /= f"{attr.metadata['dir_name']}_{val.replace('.', 'p')}"
        return path

    @property
    def to_latex(self) -> str:
        latex_str: str = f"${self._latex_name}"
        self_asdict: dict[str, Any] = asdict(self)
        for name, attr in fields_dict(type(self)).items():
            if attr.metadata.get("latex_name") is not None:
                latex_str += rf"\ {attr.metadata['latex_name']}={self_asdict[name]}"
        return latex_str + "$"

    @property
    def rng_state(self) -> dict[str, Any]:
        return self.rng.bit_generator.state

    def set_rng_state(self, rng_state: dict[str, Any] | None) -> None:
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state

    def unstructure(self) -> dict[str, Any]:
        return rmtpy_converter.unstructure(self)


@rmtpy_converter.register_structure_hook
def ensemble_structure_hook(
    src: dict[str, Any] | RandomMatrixEnsemble, _
) -> RandomMatrixEnsemble:
    if type(src) in ENSEMBLE_REGISTRY.values():
        return src

    ensemble_dict: dict[str, Any] = normalize_dict(src, ENSEMBLE_REGISTRY)
    ensemble_arguments: dict[str, Any] = ensemble_dict.pop("args")
    key: str = to_registry_key(ensemble_dict.pop("name"))
    ensemble_class: type[RandomMatrixEnsemble] = ENSEMBLE_REGISTRY[key]
    ensemble_instance: RandomMatrixEnsemble = ensemble_class(**ensemble_arguments)
    ensemble_instance.set_rng_state(src.get("rng_state"))
    return ensemble_instance


@rmtpy_converter.register_unstructure_hook
def ensemble_unstructure_hook(
    ensemble_instance: RandomMatrixEnsemble,
) -> dict[str, str | dict[str, Any]]:
    ensemble_name: str = type(ensemble_instance).__name__
    unstructured_ensemble: dict[str, Any] = asdict(ensemble_instance)
    unstructured_ensemble["name"] = to_registry_key(ensemble_name)
    unstructured_ensemble = normalize_dict(unstructured_ensemble, ENSEMBLE_REGISTRY)
    unstructured_ensemble["rng_state"] = ensemble_instance.rng_state
    return unstructured_ensemble
