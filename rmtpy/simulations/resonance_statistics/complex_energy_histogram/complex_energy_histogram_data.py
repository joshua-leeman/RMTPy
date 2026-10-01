import attrs

from ...histograms import Histogram2D

COMPLEX_ENERGY_NUM_BINS: int = 400

COMPLEX_ENERGY_REAL_SUPPORT_UNITS_SPECTRAL_RADIUS: tuple[float, float] = (-1.2, 1.2)

COMPLEX_ENERGY_WIDTH_LOG10_SUPPORT: tuple[float, float] = (-8.0, 8.0)

COMPLEX_ENERGY_WIDTH_LOG_BASE: float = 10.0


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class ComplexEnergyHistogram(Histogram2D):
    @classmethod
    def create_raw(cls) -> ComplexEnergyHistogram:
        raw_complex_energy_histogram = ComplexEnergyHistogram(
            _file_name="complex_energy_histogram",
            x_support=COMPLEX_ENERGY_REAL_SUPPORT_UNITS_SPECTRAL_RADIUS,
            x_num_bins=COMPLEX_ENERGY_NUM_BINS,
            y_log_base=COMPLEX_ENERGY_WIDTH_LOG_BASE,
            y_support=COMPLEX_ENERGY_WIDTH_LOG10_SUPPORT,
            y_num_bins=COMPLEX_ENERGY_NUM_BINS,
        )

        metadata = {"unfolding": "raw"}
        raw_complex_energy_histogram.attach_metadata(metadata)

        return raw_complex_energy_histogram

    @classmethod
    def create_unfolded(
        cls,
        *,
        file_name: str,
        unfolding: str,
        polynomial_degree: int | None = None,
    ) -> ComplexEnergyHistogram:
        unfolded_complex_energy_histogram = ComplexEnergyHistogram(
            _file_name=file_name,
            x_support=COMPLEX_ENERGY_REAL_SUPPORT_UNITS_SPECTRAL_RADIUS,
            x_num_bins=COMPLEX_ENERGY_NUM_BINS,
            y_log_base=COMPLEX_ENERGY_WIDTH_LOG_BASE,
            y_support=COMPLEX_ENERGY_WIDTH_LOG10_SUPPORT,
            y_num_bins=COMPLEX_ENERGY_NUM_BINS,
        )

        metadata: dict[str, int | str] = {"unfolding": unfolding}
        if polynomial_degree is not None:
            metadata["polynomial_degree"] = polynomial_degree
        unfolded_complex_energy_histogram.attach_metadata(metadata)

        return unfolded_complex_energy_histogram
