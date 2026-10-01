import attrs

from ....ensembles import ManyBodyEnsemble
from ...histograms import Histogram
from ...statistics import scale_support

RESONANCE_SPACING_SUPPORT_UNITS_MEAN: tuple[float, float] = (0.0, 4.0)


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class ResonanceSpacingHistogram(Histogram):
    @classmethod
    def create_raw(
        cls,
        *,
        ensemble: ManyBodyEnsemble,
    ) -> ResonanceSpacingHistogram:
        global_mean_spacing = 2 * ensemble.spectral_radius / ensemble.dimension

        raw_resonance_spacing_histogram = ResonanceSpacingHistogram(
            _file_name="resonance_spacing_histogram",
            support=scale_support(
                RESONANCE_SPACING_SUPPORT_UNITS_MEAN,
                scale=global_mean_spacing,
            ),
        )
        metadata = {
            "global_mean_spacing": global_mean_spacing,
            "unfolding": "raw",
        }
        raw_resonance_spacing_histogram.attach_metadata(metadata)

        return raw_resonance_spacing_histogram

    @classmethod
    def create_unfolded(
        cls,
        *,
        file_name: str,
        unfolding: str,
        polynomial_degree: int | None = None,
    ) -> ResonanceSpacingHistogram:
        unfolded_resonance_spacing_histogram = ResonanceSpacingHistogram(
            _file_name=file_name,
            support=RESONANCE_SPACING_SUPPORT_UNITS_MEAN,
        )
        metadata: dict[str, int | str] = {"unfolding": unfolding}
        if polynomial_degree is not None:
            metadata["polynomial_degree"] = polynomial_degree
        unfolded_resonance_spacing_histogram.attach_metadata(metadata)

        return unfolded_resonance_spacing_histogram
