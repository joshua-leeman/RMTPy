from __future__ import annotations

import ast
import inspect
from abc import ABC
from pathlib import Path
from typing import Any, ClassVar, Sequence, TypeAlias, Union

import attrs
import numpy as np
from cattrs.dispatch import StructureHook, UnstructureHook

import rmtpy.conversion

INITIALISM: str = "RME"

DTYPE_DEFAULT: np.dtype[np.complex128] = np.dtype("complex128")
DIMENSION_METADATA: dict[str, str] = {
    "dir_name": "dim",
    "latex_name": "D",
}

REGISTRY: dict[str, type[RandomMatrixEnsemble]] = {}
STRUCTURE_HOOKS: dict[str, StructureHook] = {}
UNSTRUCTURE_HOOKS: dict[str, UnstructureHook] = {}

SeedLike: TypeAlias = Union[
    None,
    bytes,
    int,
    np.random.SeedSequence,
    np.random.BitGenerator,
    np.random.Generator,
    Sequence[int],
    str,
]


def compute_complex_dtype(ens: RandomMatrixEnsemble) -> np.dtype:
    return np.dtype(ens.dtype.char.upper())


def compute_real_dtype(ens: RandomMatrixEnsemble) -> np.dtype:
    return np.dtype(ens.dtype.char.lower())


def create_random_number_generator(ens: RandomMatrixEnsemble) -> np.random.Generator:
    return np.random.default_rng(ens.seed)


def create_random_matrix_ensemble(**kwargs: Any) -> RandomMatrixEnsemble:
    return RandomMatrixEnsemble.create(kwargs)


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class RandomMatrixEnsemble(ABC):
    initialism: ClassVar[str] = INITIALISM

    dtype: np.dtype[Any] = attrs.field(
        default=DTYPE_DEFAULT,
        converter=np.dtype,
    )
    dimension: int = attrs.field(
        converter=int,
        validator=attrs.validators.gt(0),
        metadata=DIMENSION_METADATA,
    )
    seed: SeedLike = attrs.field(
        default=None,
        converter=lambda seed: (
            ast.literal_eval(seed) if isinstance(seed, str) else seed
        ),
    )

    complex_dtype: np.dtype[np.complexfloating[Any]] = attrs.field(
        default=attrs.Factory(compute_complex_dtype, takes_self=True),
        init=False,
        repr=False,
    )
    real_dtype: np.dtype[np.floating[Any]] = attrs.field(
        default=attrs.Factory(compute_real_dtype, takes_self=True),
        init=False,
        repr=False,
    )
    rng: np.random.Generator = attrs.field(
        default=attrs.Factory(create_random_number_generator, takes_self=True),
        init=False,
        repr=False,
    )

    @classmethod
    def __attrs_init_subclass__(cls) -> None:
        if inspect.isabstract(cls):
            return

        key: str = rmtpy.conversion.to_registry_key(cls.__name__)
        REGISTRY[key] = cls
        STRUCTURE_HOOKS[key] = rmtpy.conversion.CONVERTER.get_structure_hook(cls)
        UNSTRUCTURE_HOOKS[key] = rmtpy.conversion.CONVERTER.get_unstructure_hook(cls)

    @classmethod
    def create(cls, src: dict[str, Any] | RandomMatrixEnsemble) -> RandomMatrixEnsemble:
        return rmtpy.conversion.CONVERTER.structure(src, cls)

    @property
    def latex_name(self) -> str:
        return f"\\textrm{{{self.initialism}}}"

    @property
    def token_name(self) -> str:
        return type(self).initialism

    @property
    def as_latex(self) -> str:
        return rmtpy.conversion.to_latex(self, type(self).latex_name)

    @property
    def as_path(self) -> Path:
        return rmtpy.conversion.to_path(self, Path(type(self).token_name))

    @property
    def rng_state(self) -> dict[str, Any]:
        return self.rng.bit_generator.state

    def set_rng_state(self, rng_state: dict[str, Any] | None) -> None:
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state

    def unstructure(self) -> dict[str, Any]:
        return rmtpy.conversion.CONVERTER.unstructure(self)


@rmtpy.conversion.CONVERTER.register_structure_hook
def structure_hook_for_ensemble(src: dict | Any, _) -> RandomMatrixEnsemble:
    if type(src) in REGISTRY.values():
        return src

    ens_dict: dict[str, Any] = rmtpy.conversion.normalize_dict(src, REGISTRY)
    ens_args: dict[str, Any] = ens_dict.pop("args")
    key: str = rmtpy.conversion.to_registry_key(ens_dict.pop("name"))
    ens_cls: type[RandomMatrixEnsemble] = REGISTRY[key]
    ens_inst: RandomMatrixEnsemble = ens_cls(**ens_args)
    ens_inst.set_rng_state(src.get("rng_state"))
    return ens_inst


@rmtpy.conversion.CONVERTER.register_unstructure_hook
def unstructure_hook_for_ensemble(ens: RandomMatrixEnsemble) -> dict[str, Any]:
    args: dict[str, Any] = {}
    for name, attr in attrs.fields_dict(type(ens)).items():
        if attr.init:
            args[name] = rmtpy.conversion.CONVERTER.unstructure(getattr(ens, name))

    return {
        "name": rmtpy.conversion.to_registry_key(type(ens).__name__),
        "args": args,
        "rng_state": ens.rng_state,
    }
