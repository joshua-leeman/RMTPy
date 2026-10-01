import attrs

from ...histograms import Histogram

WIDTH_LOG10_SUPPORT: tuple[float, float] = (-4.0, 4.0)

WIDTH_LOG_BASE: float = 10.0


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class WidthHistogram(Histogram):
    @classmethod
    def create_raw(cls) -> WidthHistogram:
        raw_width_histogram = WidthHistogram(
            _file_name="width_histogram",
            log_base=WIDTH_LOG_BASE,
            support=WIDTH_LOG10_SUPPORT,
        )

        metadata = {"unfolding": "raw"}
        raw_width_histogram.attach_metadata(metadata)

        return raw_width_histogram

    @classmethod
    def create_unfolded(
        cls,
        *,
        file_name: str,
        unfolding: str,
        polynomial_degree: int | None = None,
    ) -> WidthHistogram:
        unfolded_width_histogram = WidthHistogram(
            _file_name=file_name,
            log_base=WIDTH_LOG_BASE,
            support=WIDTH_LOG10_SUPPORT,
        )

        metadata: dict[str, int | str] = {"unfolding": unfolding}
        if polynomial_degree is not None:
            metadata["polynomial_degree"] = polynomial_degree
        unfolded_width_histogram.attach_metadata(metadata)

        return unfolded_width_histogram
