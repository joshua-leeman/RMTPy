from __future__ import annotations

from collections.abc import Iterator

import attrs
import numpy as np

import rmtpy.density

from .histogram import Histogram
from .observable import Observable


def iter_objects(value: object, cls: type) -> Iterator[object]:
    if isinstance(value, cls):
        yield value

    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_objects(item, cls)

    elif isinstance(value, list | tuple):
        for item in value:
            yield from iter_objects(item, cls)

    elif attrs.has(type(value)) and type(value).__module__.startswith(
        "rmtpy.simulations"
    ):
        for field in attrs.fields(type(value)):
            yield from iter_objects(getattr(value, field.name), cls)


def iter_observables(value: object) -> Iterator[Observable]:
    yield from iter_objects(value, Observable)


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class CoefficientHistogramOutputs:
    by_degree: tuple[Observable[Histogram], ...]

    def add(
        self, values: np.ndarray, *, density: rmtpy.density.DensityModel
    ) -> np.ndarray:
        coeffs = density.compute_variate_coeffs(values)
        for observable, coeff in zip(self.by_degree, coeffs[1:], strict=True):
            observable.data.add_histogram_contribution(coeff)

        return coeffs
