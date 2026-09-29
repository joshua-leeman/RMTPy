import attrs

from ....density import DensityModel
from ....simulations.statistics import scale_support
from ...histograms import Histogram

UNFOLDED_LEVEL_SUPPORT_UNITS_DIMENSION: tuple[float, float] = (-1.2, 1.2)


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class SpectralHistogram(Histogram):
    @classmethod
    def create_raw(
        cls,
        *,
        spectral_density: DensityModel,
    ) -> SpectralHistogram:
        raw_spectral_histogram = SpectralHistogram(
            file_name="spectral_histogram_data",
            support=spectral_density.plot_range,
        )

        metadata = {"unfolding": "raw"}
        raw_spectral_histogram.attach_metadata(metadata)

        return raw_spectral_histogram

    @classmethod
    def create_unfolded(
        cls,
        *,
        file_name_prefix: str,
        dimension: int,
        unfolding: str,
        polynomial_degree: int | None = None,
    ) -> SpectralHistogram:
        unfolded_level_support = scale_support(
            UNFOLDED_LEVEL_SUPPORT_UNITS_DIMENSION,
            scale=dimension,
        )

        unfolded_spectral_histogram = SpectralHistogram(
            file_name=f"{file_name_prefix}_data",
            support=unfolded_level_support,
        )

        metadata: dict[str, int | str] = {"unfolding": unfolding}
        if polynomial_degree is not None:
            metadata["polynomial_degree"] = polynomial_degree
        unfolded_spectral_histogram.attach_metadata(metadata)

        return unfolded_spectral_histogram
