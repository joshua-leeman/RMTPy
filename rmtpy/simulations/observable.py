from collections.abc import Callable
from pathlib import Path
from typing import Any

import attrs

from .data import Data
from .plot import Plot


def validate_plot_cls(plot_cls: type[Plot]) -> None:
    if not issubclass(plot_cls, Plot):
        raise ValueError("`plot_cls` must be a subclass of `Plot`")


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class Observable:
    data: Data = attrs.field(
        validator=attrs.validators.instance_of(Data),
        repr=False,
    )
    finalize: Callable[[Data], None] | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(attrs.validators.is_callable()),
        repr=False,
    )
    plot_cls: type[Plot] | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(
            lambda _, __, plot_cls: validate_plot_cls(plot_cls)
        ),
        repr=False,
    )

    plot: Plot | None = attrs.field(
        default=None,
        init=False,
        repr=False,
    )

    @property
    def metadata(self) -> dict[str, Any]:
        return self.data.metadata

    def initialize_plot(self) -> None:
        if self.plot_cls is not None:
            object.__setattr__(self, "plot", self.plot_cls(data=self.data))

    def calculate_statistics(self) -> None:
        if self.finalize is not None:
            self.finalize(self.data)

    def save_data(self, out_dir: Path) -> None:
        subdir_name = self.data.file_name.replace("_data", "")
        out_path = out_dir / subdir_name / f"{self.data.file_name}.npz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.data.save(out_path)

    def save_plot(self, out_dir: Path) -> None:
        if self.plot is None:
            self.initialize_plot()
        if self.plot is not None:
            subdir_name = self.data.file_name.replace("_data", "")
            self.plot.plot(path=out_dir / subdir_name)
