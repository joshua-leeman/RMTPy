from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any

import attrs
import numpy as np

import rmtpy.conversion
from rmtpy.conversion import RMT_CONVERTER

from .data import Data
from .observable import Observable


def normalize_metadata_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return normalize_metadata_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: normalize_metadata_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [normalize_metadata_value(item) for item in value]
    return value


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class Simulation(ABC):
    """Experiment coordinator whose output bundle owns the mutable accumulators."""

    metadata: dict[str, Any] = attrs.field(
        factory=dict,
        init=False,
        repr=False,
    )

    def __attrs_post_init__(self) -> None:
        self.populate_metadata()

        for observable in self.iter_observables():
            observable.metadata.update({"simulation": self.metadata.copy()})

    @property
    def path_name(self) -> str:
        return rmtpy.conversion.insert_underscores(type(self).__name__).lower()

    @property
    def to_path(self) -> Path:
        return rmtpy.conversion.to_path(self, root=Path(self.path_name))

    def populate_metadata(self) -> None:
        self.metadata["name"] = type(self).__name__
        self.metadata["args"] = {
            name: self._unstructure_argument(getattr(self, name))
            for name, field in attrs.fields_dict(type(self)).items()
            if field.init
        }

    @staticmethod
    def _unstructure_argument(value: Any) -> Any:
        return normalize_metadata_value(RMT_CONVERTER.unstructure(value))

    def iter_observables(self) -> Iterator[Observable]:
        outputs = getattr(self, "outputs", None)
        if outputs is None or not hasattr(outputs, "iter_observables"):
            raise NotImplementedError(
                f"{type(self).__name__} must define an output bundle with "
                "`iter_observables()`."
            )
        yield from outputs.iter_observables()

    def find_observables(
        self,
        file_name: str | None = None,
        **metadata: Any,
    ) -> tuple[Observable, ...]:
        """Return observables matching a data file name and metadata values."""
        normalized_name = None
        if file_name is not None:
            normalized_name = file_name.removesuffix("_data")

        return tuple(
            observable
            for observable in self.iter_observables()
            if (
                normalized_name is None
                or observable.data.file_name.removesuffix("_data") == normalized_name
            )
            and all(
                observable.metadata.get(key) == value for key, value in metadata.items()
            )
        )

    def get_observable(
        self,
        file_name: str | None = None,
        **metadata: Any,
    ) -> Observable:
        """Return one observable, raising when the selection is absent or ambiguous."""
        matches = self.find_observables(file_name, **metadata)
        if len(matches) != 1:
            selection = {"file_name": file_name, **metadata}
            raise LookupError(
                f"Expected one observable matching {selection}, found {len(matches)}."
            )
        return matches[0]

    def get_data(self, file_name: str | None = None, **metadata: Any) -> Data:
        """Return the data carried by one matching observable."""
        return self.get_observable(file_name, **metadata).data

    def observable_output_path(self, observable: Observable) -> Path:
        return Path()

    @abstractmethod
    def realize_monte_carlo_simulation(self) -> None:
        raise NotImplementedError()

    def calculate_statistics(self) -> None:
        for observable in self.iter_observables():
            observable.calculate_statistics()

    def save_metadata(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "metadata.json", "w") as file:
            json.dump(self.metadata, file, indent=4, default=str)

    def save_data(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        self.save_metadata(out_dir)

        for observable in self.iter_observables():
            observable.save_data(out_dir / self.observable_output_path(observable))

    def save_plots(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        simulation_args = deepcopy(self.metadata["args"])
        for observable in self.iter_observables():
            observable.save_plot(
                out_dir / self.observable_output_path(observable),
                simulation_args=simulation_args,
            )

    def run(self, out_dir: str | Path = "output") -> None:
        self.realize_monte_carlo_simulation()
        self.calculate_statistics()

        out_dir = Path(out_dir)
        base_dir = out_dir / self.to_path
        base_dir.mkdir(parents=True, exist_ok=True)

        self.save_data(out_dir=base_dir)
        self.save_plots(out_dir=base_dir)
