import math
from typing import ClassVar

import attrs
import numpy as np

from .base_data import Data
from .histograms import NUM_BINS, Histogram

POLYNOMIAL_DEGREE_MIN: int = 1
POLYNOMIAL_DEGREE_STEP: int = 1

REALIZATIONS_METADATA: dict[str, str] = {
    "dir_name": "realizs",
    "latex_name": "R",
}

LOG_D_TIME_SUPPORT: tuple[float, float] = (-0.5, 1.5)

LOG_D_UNFOLDED_TIME_SUPPORT: tuple[float, float] = (-1.5, 0.5)

COEFFICIENT_GRID_POLICY: str = "configuration_v1"


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class CoefficientsHistogram(Histogram):
    coefficient_name: ClassVar[str]

    underflow: int = attrs.field(
        default=0,
        converter=int,
        validator=attrs.validators.ge(0),
        metadata={"archive_optional": True},
        repr=False,
    )
    overflow: int = attrs.field(
        default=0,
        converter=int,
        validator=attrs.validators.ge(0),
        metadata={"archive_optional": True},
        repr=False,
    )

    @classmethod
    def create(
        cls,
        *,
        degree: int,
        dimension: int,
    ) -> CoefficientsHistogram:
        degree = int(degree)
        dimension = int(dimension)
        if degree < POLYNOMIAL_DEGREE_MIN:
            raise ValueError("Coefficient polynomial degree must be positive.")
        if dimension < 1:
            raise ValueError("Coefficient histogram dimension must be positive.")

        half_width = (degree + 1) / math.sqrt(dimension)
        num_bins = max(NUM_BINS, math.ceil(math.sqrt(dimension)))
        histogram = cls(
            _file_name=f"{cls.coefficient_name}_coeff_{degree}_histogram",
            support=(-half_width, half_width),
            num_bins=num_bins,
        )
        histogram.attach_metadata(
            {
                "degree": degree,
                "unfolding": "raw",
                "grid_policy": COEFFICIENT_GRID_POLICY,
                "dimension": dimension,
            }
        )
        return histogram

    def add_histogram_contribution(
        self,
        data: np.ndarray[tuple[int], np.dtype[np.floating]],
        /,
    ) -> None:
        values = np.asarray(data)
        if not np.all(np.isfinite(values)):
            raise ValueError("Coefficient samples must be finite.")

        object.__setattr__(
            self,
            "underflow",
            self.underflow + int(np.count_nonzero(values < self.bins[0])),
        )
        object.__setattr__(
            self,
            "overflow",
            self.overflow + int(np.count_nonzero(values >= self.bins[-1])),
        )
        super().add_histogram_contribution(values)

    def add_contribution(self, contribution: Data, /) -> None:
        if not isinstance(contribution, CoefficientsHistogram):
            raise TypeError("Coefficient histogram contribution is malformed.")
        if contribution.metadata.get("grid_policy") != COEFFICIENT_GRID_POLICY:
            raise ValueError(
                f"Coefficient histogram `{contribution._file_name}` predates the "
                + "deterministic aggregation grid and cannot be aggregated exactly."
            )

        super().add_contribution(contribution)
        object.__setattr__(self, "underflow", self.underflow + contribution.underflow)
        object.__setattr__(self, "overflow", self.overflow + contribution.overflow)


def nearest_neighbor_spacings(
    values: np.ndarray[tuple[int], np.dtype[np.floating]],
    /,
    *,
    degeneracy: int = 1,
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    spacings = np.diff(np.sort(values))
    if degeneracy > 1:
        spacings = np.repeat(spacings[1::degeneracy], degeneracy)

    return spacings


def scale_support(
    support: tuple[float, float],
    /,
    *,
    scale: float,
) -> tuple[float, float]:
    return scale * support[0], scale * support[1]


def truncated_polynomial_degree_range(*, max_degree: int) -> range:
    return range(POLYNOMIAL_DEGREE_MIN, max_degree + 1, POLYNOMIAL_DEGREE_STEP)
