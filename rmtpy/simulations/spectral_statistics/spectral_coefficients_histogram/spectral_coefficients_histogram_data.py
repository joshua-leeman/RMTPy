import attrs

from ....simulations.histograms import Histogram

_INITIAL_SPECTRAL_COEFFICIENT_SUPPORT: tuple[float, float] = (-0.2, 0.2)


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class SpectralCoefficientsHistogram(Histogram):
    @classmethod
    def create(
        cls,
        *,
        degree: int,
    ) -> SpectralCoefficientsHistogram:
        coefficient_histogram = SpectralCoefficientsHistogram(
            _file_name=f"spectral_coeff_{degree}_histogram",
            support=_INITIAL_SPECTRAL_COEFFICIENT_SUPPORT,
        )
        metadata = {"degree": degree, "unfolding": "raw"}
        coefficient_histogram.attach_metadata(metadata)

        return coefficient_histogram
