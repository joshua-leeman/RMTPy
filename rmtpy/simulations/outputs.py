from __future__ import annotations

from collections.abc import Iterator

import attrs
import numpy as np

import rmtpy.density

from .histogram import Histogram
from .observable import Observable


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class CoefficientHistogramOutputs:
    """Typed, ordered histograms for density coefficients by degree."""

    by_degree: tuple[Observable[Histogram], ...]

    def iter_observables(self) -> Iterator[Observable]:
        yield from self.by_degree

    def add(
        self, values: np.ndarray, *, density: rmtpy.density.DensityModel
    ) -> np.ndarray:
        coeffs = density.compute_variate_coeffs(values)
        for observable, coeff in zip(self.by_degree, coeffs[1:], strict=True):
            observable.data.add_histogram_contribution(coeff)

        return coeffs
