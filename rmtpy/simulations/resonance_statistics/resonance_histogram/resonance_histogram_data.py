import attrs

from ....density import DensityModel
from ...histograms import Histogram
from ...statistics import scale_support

UNFOLDED_RESONANCE_SUPPORT_UNITS_DIMENSION: tuple[float, float] = (-1.2, 1.2)


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class ResonanceHistogram(Histogram):
    @classmethod
    def create_raw(
        cls,
        *,
        resonance_density: DensityModel,
    ) -> ResonanceHistogram:
        raw_resonance_histogram = ResonanceHistogram(
            _file_name="resonance_histogram",
            support=resonance_density.plot_range,
        )

        metadata = {"unfolding": "raw"}
        raw_resonance_histogram.attach_metadata(metadata)

        return raw_resonance_histogram

    @classmethod
    def create_unfolded(
        cls,
        *,
        file_name: str,
        dimension: int,
        unfolding: str,
        polynomial_degree: int | None = None,
    ) -> ResonanceHistogram:
        unfolded_resonance_support = scale_support(
            UNFOLDED_RESONANCE_SUPPORT_UNITS_DIMENSION,
            scale=dimension,
        )

        unfolded_resonance_histogram = ResonanceHistogram(
            _file_name=file_name,
            support=unfolded_resonance_support,
        )

        metadata: dict[str, int | str] = {"unfolding": unfolding}
        if polynomial_degree is not None:
            metadata["polynomial_degree"] = polynomial_degree
        unfolded_resonance_histogram.attach_metadata(metadata)

        return unfolded_resonance_histogram
