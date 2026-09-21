from pathlib import Path

import attrs


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class Data:
    file_name: str = attrs.field(
        default="simulation_data",
        converter=str,
    )
    metadata: dict[str, object] = attrs.field(
        factory=dict,
        converter=dict,
        repr=False,
    )

    def attach_metadata(self, metadata: dict[str, int | float | str]) -> None:
        self.metadata.update(metadata)

    def build_data_path(self) -> Path:
        stem = self.file_name.removesuffix("_data")
        return Path(stem) / f"{self.file_name}.npz"
