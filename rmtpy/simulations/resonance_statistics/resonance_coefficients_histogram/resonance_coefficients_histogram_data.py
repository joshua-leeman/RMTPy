import attrs

from ...histograms import Histogram

RESONANCE_COEFFICIENT_SUPPORT: tuple[float, float] = (-0.2, 0.2)


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class ResonanceCoefficientsHistogram(Histogram):
    @classmethod
    def create(
        cls,
        *,
        degree: int,
    ) -> ResonanceCoefficientsHistogram:
        coefficient_histogram = ResonanceCoefficientsHistogram(
            _file_name=f"resonance_coeff_{degree}_histogram",
            support=RESONANCE_COEFFICIENT_SUPPORT,
        )
        metadata = {"degree": degree, "unfolding": "raw"}
        coefficient_histogram.attach_metadata(metadata)

        return coefficient_histogram
