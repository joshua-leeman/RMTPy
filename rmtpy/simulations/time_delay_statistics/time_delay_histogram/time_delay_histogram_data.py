from typing import cast

import attrs
import numpy as np
from scipy.special import jn_zeros

from ....ensembles import ManyBodyEnsemble
from ...histograms import Histogram
from ...statistics import LOG_D_TIME_SUPPORT, LOG_D_UNFOLDED_TIME_SUPPORT

TIME_DELAY_NUM_BINS: int = 100


def _scale_logarithmic_support(
    support: tuple[float, float],
    *,
    log_base: float,
    scale: float,
) -> tuple[float, float]:
    logarithmic_scale = cast(float, np.log(scale) / np.log(log_base))
    return support[0] + logarithmic_scale, support[1] + logarithmic_scale


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class TimeDelayHistogram(Histogram):
    @classmethod
    def create_raw(
        cls,
        *,
        ensemble: ManyBodyEnsemble,
        energy_index: int,
        energy: float,
    ) -> TimeDelayHistogram:
        first_bessel_zero = cast(float, jn_zeros(1, 1)[0])
        scale = first_bessel_zero / ensemble.spectral_radius
        support = _scale_logarithmic_support(
            LOG_D_TIME_SUPPORT,
            log_base=ensemble.dimension,
            scale=scale,
        )

        raw_time_delay_histogram = TimeDelayHistogram(
            _file_name=f"time_delay_energy_{energy_index}_histogram",
            support=support,
            log_base=ensemble.dimension,
            num_bins=TIME_DELAY_NUM_BINS,
        )

        metadata = {
            "energy_index": energy_index,
            "energy": energy,
            "scale": scale,
            "unfolding": "raw",
        }
        raw_time_delay_histogram.attach_metadata(metadata)

        return raw_time_delay_histogram

    @classmethod
    def create_unfolded(
        cls,
        *,
        ensemble: ManyBodyEnsemble,
        energy_index: int,
        energy: float,
        unfolding: str,
        polynomial_degree: int | None = None,
    ) -> TimeDelayHistogram:
        scale = 2 * np.pi
        support = _scale_logarithmic_support(
            LOG_D_UNFOLDED_TIME_SUPPORT,
            log_base=ensemble.dimension,
            scale=scale,
        )
        suffix = f"_degree_{polynomial_degree}" if polynomial_degree is not None else ""

        unfolded_time_delay_histogram = TimeDelayHistogram(
            _file_name=(
                f"time_delay_energy_{energy_index}_histogram_"
                + f"{unfolding}_unfolded{suffix}"
            ),
            support=support,
            log_base=ensemble.dimension,
            num_bins=TIME_DELAY_NUM_BINS,
        )

        metadata: dict[str, float | int | str] = {
            "energy_index": energy_index,
            "energy": energy,
            "scale": scale,
            "unfolding": unfolding,
        }
        if polynomial_degree is not None:
            metadata["polynomial_degree"] = polynomial_degree
        unfolded_time_delay_histogram.attach_metadata(metadata)

        return unfolded_time_delay_histogram
