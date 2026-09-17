import ast
import inspect
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import ClassVar, cast

import attrs
import numpy as np

from ..conversion import (
    RMT_CONVERTER,
    SourceDict,
    canonicalize_source_dict,
    to_key_of_registry,
    to_latex,
    to_path,
)

type SeedLike = (
    None
    | bytes
    | int
    | np.random.SeedSequence
    | np.random.BitGenerator
    | np.random.Generator
    | Sequence[int]
)

INITIALISM: str = "RME"

DTYPE: np.dtype[np.generic] = np.dtype("complex128")

DIMENSION_METADATA: dict[str, str] = {
    "dir_name": "dim",
    "latex_name": "D",
}

REGISTRY: dict[str, type[attrs.AttrsInstance]] = {}


def _compute_complex_dtype(ensemble: RandomMatrixEnsemble) -> np.dtype[np.complex128]:
    return np.dtype(ensemble.dtype.char.upper())


def _compute_real_dtype(ensemble: RandomMatrixEnsemble) -> np.dtype[np.float64]:
    return np.dtype(ensemble.dtype.char.lower())


def _compute_seed(seed: str | SeedLike) -> SeedLike:
    return ast.literal_eval(seed) if isinstance(seed, str) else seed


def _build_random_number_generator(
    ensemble: RandomMatrixEnsemble,
) -> np.random.Generator:
    return np.random.default_rng(ensemble.seed)


def _structure_hook_for_ensemble(
    src: SourceDict | RandomMatrixEnsemble,
    _: object,
) -> RandomMatrixEnsemble:
    if isinstance(src, dict):
        ensemble_dict = canonicalize_source_dict(src, registry=REGISTRY)

        ensemble_type = ensemble_dict["type"]
        if not isinstance(ensemble_type, str):
            raise TypeError("Configuration `type` must be a string.")

        parameters = ensemble_dict["parameters"]
        if not isinstance(parameters, dict):
            raise TypeError("Configuration `parameters` must be a dictionary.")

        key = to_key_of_registry(ensemble_type)
        ensemble_factory = cast(Callable[..., RandomMatrixEnsemble], REGISTRY[key])
        return ensemble_factory(**parameters)

    return src


def _unstructure_hook_for_ensemble(
    ensemble: RandomMatrixEnsemble,
) -> SourceDict:
    fields = cast(dict[str, attrs.Attribute[object]], attrs.fields_dict(type(ensemble)))
    parameters = {
        name: RMT_CONVERTER.unstructure(getattr(ensemble, name))
        for name, attr in fields.items()
        if attr.init
    }

    return {
        "type": type(ensemble).__name__,
        "parameters": parameters,
    }


def _register_ensemble_hooks(
    ensemble_cls: type[RandomMatrixEnsemble],
) -> type[RandomMatrixEnsemble]:
    RMT_CONVERTER.register_structure_hook(ensemble_cls, _structure_hook_for_ensemble)
    RMT_CONVERTER.register_unstructure_hook(ensemble_cls, _unstructure_hook_for_ensemble)

    return ensemble_cls


@_register_ensemble_hooks
@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class RandomMatrixEnsemble:
    initialism: ClassVar[str] = INITIALISM

    dimension: int = attrs.field(
        validator=(
            attrs.validators.instance_of(int),
            attrs.validators.gt(0),
        ),
        metadata=DIMENSION_METADATA,
    )
    dtype: np.dtype[np.generic] = attrs.field(
        default=DTYPE,
        converter=np.dtype,
    )
    seed: SeedLike = attrs.field(
        default=None,
        converter=_compute_seed,
    )

    complex_dtype: np.dtype[np.complex128] = attrs.field(
        default=attrs.Factory(_compute_complex_dtype, takes_self=True),
        init=False,
        repr=False,
    )
    real_dtype: np.dtype[np.float64] = attrs.field(
        default=attrs.Factory(_compute_real_dtype, takes_self=True),
        init=False,
        repr=False,
    )
    rng: np.random.Generator = attrs.field(
        default=attrs.Factory(_build_random_number_generator, takes_self=True),
        init=False,
        repr=False,
    )

    @classmethod
    def __attrs_init_subclass__(cls) -> None:
        if inspect.isabstract(cls):
            return

        key = to_key_of_registry(cls.__name__)
        REGISTRY[key] = cls

    @classmethod
    def create(
        cls,
        src: SourceDict | RandomMatrixEnsemble,
    ) -> RandomMatrixEnsemble:
        return RMT_CONVERTER.structure(src, cls)

    @property
    def latex_name(self) -> str:
        return f"\\text{{{type(self).initialism}}}"

    @property
    def token_name(self) -> str:
        return type(self).initialism

    @property
    def to_latex(self) -> str:
        return to_latex(self, latex_name=self.latex_name)

    @property
    def to_path(self) -> Path:
        return to_path(self, root=Path(self.token_name))

    @property
    def rng_state(self) -> dict[str, object]:
        return dict(self.rng.bit_generator.state)
