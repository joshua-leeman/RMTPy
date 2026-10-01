import attrs

from ...histograms import Histogram

PARTIAL_WIDTH_LOG10_SUPPORT: tuple[float, float] = (-5.0, 2.0)

PARTIAL_WIDTH_NUM_BINS: int = 60

WIDTH_LOG_BASE: float = 10.0


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class PartialWidthHistogram(Histogram):
    @classmethod
    def create(
        cls,
        *,
        state_index: int,
        channel_index: int,
    ) -> PartialWidthHistogram:
        partial_width_histogram = PartialWidthHistogram(
            _file_name=(
                f"partial_width_state_{state_index}_channel_{channel_index}_histogram"
            ),
            log_base=WIDTH_LOG_BASE,
            support=PARTIAL_WIDTH_LOG10_SUPPORT,
            num_bins=PARTIAL_WIDTH_NUM_BINS,
        )

        metadata = {
            "index": [state_index, channel_index],
            "average_width": 0.0,
            "unfolding": "raw",
        }
        partial_width_histogram.attach_metadata(metadata)

        return partial_width_histogram
