import attrs

from ...histograms import MeanScaledHistogram

TOTAL_WIDTH_LOG10_SUPPORT: tuple[float, float] = (-2.0, 3.0)

TOTAL_WIDTH_NUM_BINS: int = 100

WIDTH_LOG_BASE: float = 10.0


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class TotalWidthHistogram(MeanScaledHistogram):
    @classmethod
    def create(
        cls,
        *,
        state_index: int,
    ) -> TotalWidthHistogram:
        total_width_histogram = TotalWidthHistogram(
            _file_name=f"total_width_state_{state_index}_histogram",
            log_base=WIDTH_LOG_BASE,
            support=TOTAL_WIDTH_LOG10_SUPPORT,
            num_bins=TOTAL_WIDTH_NUM_BINS,
        )

        metadata = {
            "index": [state_index],
            "average_width": 0.0,
            "unfolding": "raw",
        }
        total_width_histogram.attach_metadata(metadata)

        return total_width_histogram
