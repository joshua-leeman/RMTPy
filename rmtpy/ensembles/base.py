from __future__ import annotations

import ast
import inspect
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, TypeAlias

import attrs
import numpy as np

import rmtpy.conversion
from rmtpy.conversion import RMT_CONVERTER

INITIALISM: str = "RME"

DTYPE: np.dtype[np.complex128] = np.dtype("complex128")

DIMENSION_METADATA: dict[str, str] = {
    "dir_name": "dim",
    "latex_name": "D",
}

REGISTRY: dict[str, type[RandomMatrixEnsemble]] = {}

SeedLike: TypeAlias = (
    None
    | bytes
    | int
    | np.random.SeedSequence
    | np.random.BitGenerator
    | np.random.Generator
    | Sequence[int]
    | str
)


def compute_complex_dtype(ens: RandomMatrixEnsemble) -> np.dtype:
    return np.dtype(ens.dtype.char.upper())


def compute_real_dtype(ens: RandomMatrixEnsemble) -> np.dtype:
    return np.dtype(ens.dtype.char.lower())


def create_random_number_generator(ens: RandomMatrixEnsemble) -> np.random.Generator:
    return np.random.default_rng(ens.seed)


def create_random_matrix_ensemble(**kwargs: Any) -> RandomMatrixEnsemble:
    return RandomMatrixEnsemble.create(kwargs)


def structure_hook_for_ensemble(src: dict | Any, _) -> RandomMatrixEnsemble:
    if type(src) in REGISTRY.values():
        return src

    ens_dict = rmtpy.conversion.normalize_dict(src, registry=REGISTRY)
    ens_args = ens_dict.pop("args")

    key = rmtpy.conversion.to_registry_key(ens_dict.pop("name"))
    ens_cls = REGISTRY[key]

    ens_inst = ens_cls(**ens_args)
    ens_inst.set_rng_state(src.get("rng_state"))
    return ens_inst


def unstructure_hook_for_ensemble(ens: RandomMatrixEnsemble) -> dict[str, Any]:
    arguments = {
        name: RMT_CONVERTER.unstructure(getattr(ens, name))
        for name, attr in attrs.fields_dict(type(ens)).items()
        if attr.init
    }

    return {
        "name": rmtpy.conversion.to_registry_key(type(ens).__name__),
        "args": arguments,
        "rng_state": ens.rng_state,
    }


def register_ensemble_hooks(
    ens_cls: type[RandomMatrixEnsemble],
) -> type[RandomMatrixEnsemble]:
    RMT_CONVERTER.register_structure_hook(ens_cls, structure_hook_for_ensemble)
    RMT_CONVERTER.register_unstructure_hook(ens_cls, unstructure_hook_for_ensemble)

    return ens_cls


@register_ensemble_hooks
@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class RandomMatrixEnsemble:
    """Common seeded configuration and conversion for random-matrix ensembles."""

    initialism: ClassVar[str] = INITIALISM

    dimension: int = attrs.field(
        converter=int,
        validator=attrs.validators.gt(0),
        metadata=DIMENSION_METADATA,
    )
    dtype: np.dtype[Any] = attrs.field(
        default=DTYPE,
        converter=np.dtype,
    )
    seed: SeedLike = attrs.field(
        default=None,
        converter=lambda seed: ast.literal_eval(seed) if isinstance(seed, str) else seed,
    )

    complex_dtype: np.dtype[np.complexfloating[Any, Any]] = attrs.field(
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

        key = rmtpy.conversion.to_registry_key(cls.__name__)
        REGISTRY[key] = cls

    @classmethod
    def create(cls, src: dict[str, Any] | RandomMatrixEnsemble) -> RandomMatrixEnsemble:
        return RMT_CONVERTER.structure(src, cls)

    @property
    def latex_name(self) -> str:
        return f"\\text{{{type(self).initialism}}}"

    @property
    def token_name(self) -> str:
        return type(self).initialism

    @property
    def to_latex(self) -> str:
        return rmtpy.conversion.to_latex(self, latex_name=self.latex_name)

    @property
    def to_path(self) -> Path:
        return rmtpy.conversion.to_path(self, root=Path(self.token_name))

    @property
    def rng_state(self) -> dict[str, Any]:
        return self.rng.bit_generator.state

    def set_rng_state(self, rng_state: dict[str, Any] | None) -> None:
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state

    def unstructure(self) -> dict[str, Any]:
        return RMT_CONVERTER.unstructure(self)
