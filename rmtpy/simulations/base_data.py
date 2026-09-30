from collections.abc import Mapping
from pathlib import Path
from typing import cast

import attrs
import numpy as np

from rmtpy.conversion import numpy_savez_serializer


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class Data:
    metadata: dict[str, object] = attrs.field(
        factory=dict,
        converter=dict,
        repr=False,
    )

    _file_name: str = attrs.field(
        default="simulation",
        converter=str,
    )

    @property
    def to_path(self) -> Path:
        return Path(self._file_name) / Path(f"{self._file_name}_data.npz")

    def attach_metadata(self, new_metadata: Mapping[str, object], /) -> None:
        self.metadata.update(new_metadata)

    def save(self, *, directory: str | Path | None = None) -> None:
        path = Path(directory) if directory is not None else self.to_path
        if path.suffix == "":
            path /= self.to_path

        path.parent.mkdir(parents=True, exist_ok=True)

        self_asdict = cast(
            dict[str, np.ndarray[tuple[int, ...], np.dtype[np.generic]]],
            attrs.asdict(self, value_serializer=numpy_savez_serializer),
        )
        np.savez_compressed(path, allow_pickle=False, **self_asdict)
