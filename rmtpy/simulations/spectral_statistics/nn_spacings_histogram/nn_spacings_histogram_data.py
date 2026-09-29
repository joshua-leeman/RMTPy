import attrs

from ....ensembles import ManyBodyEnsemble
from ....simulations.statistics import scale_support
from ...histograms import Histogram

SPACING_SUPPORT_UNITS_MEAN: tuple[float, float] = (0.0, 4.0)


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class SpacingsHistogram(Histogram):
    @classmethod
    def create_raw_histogram(
        cls,
        *,
        ensemble: ManyBodyEnsemble,
    ) -> SpacingsHistogram:
        global_mean_spacing = 2 * ensemble.spectral_radius / ensemble.dimension

        raw_spacings_histogram = SpacingsHistogram(
            file_name="spacings_histogram_data",
            support=scale_support(
                SPACING_SUPPORT_UNITS_MEAN,
                scale=global_mean_spacing,
            ),
        )
        metadata = {"global_mean_spacing": global_mean_spacing}
        raw_spacings_histogram.attach_metadata(metadata)

        return raw_spacings_histogram

    @classmethod
    def create_unfolded_histogram(
        cls,
        *,
        file_name_prefix: str,
        unfolding: str,
        degree: int | None = None,
    ) -> SpacingsHistogram:
        unfolded_spacings_histogram = SpacingsHistogram(
            file_name=f"{file_name_prefix}_data",
            support=SPACING_SUPPORT_UNITS_MEAN,
        )
        metadata: dict[str, int | str] = {"unfolding": unfolding}
        if degree is not None:
            metadata["degree"] = degree
        unfolded_spacings_histogram.attach_metadata(metadata)

        return unfolded_spacings_histogram
