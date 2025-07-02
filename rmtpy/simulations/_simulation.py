# rmtpy/simulations/_simulation.py

# Postponed evaluation of annotations
from __future__ import annotations

# Standard library imports
import inspect
import os
import re
import shutil
from abc import ABC
from pathlib import Path
from typing import Any

# Third-party imports
import numpy as np
from attrs import asdict, field, fields_dict, frozen
from attrs.validators import gt

# Local imports
from ..ensembles import Ensemble


# -------------------------------
# Monte Carlo Simulation Registry
# -------------------------------
SIMULATION_REGISTRY: dict[str, type[Simulation]] = {}


# -------------------------
# Monte Carlo Data Registry
# -------------------------
DATA_REGISTRY: dict[str, type[Data]] = {}


# --------------------------
# Simulation Data Base Class
# --------------------------
@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class Data(ABC):
    # Simulation metadata
    simulation: dict[str, str | dict[str, Any]] = field(init=False)

    @classmethod
    def __attrs_init_subclass__(cls) -> None:
        """Register concrete subclasses in the data registry."""
        # Include only concrete classes in registry
        if not inspect.isabstract(cls):
            # Convert data class to registry key format
            data_key = re.sub(r"_", "", cls.__name__).lower().replace("data", "")

            # Normalize class name to registry key format
            DATA_REGISTRY[data_key] = cls

    @classmethod
    def load(cls, path: str | Path) -> Data:
        """Load simulation data from a serialized file."""
        # Ensure path is a Path object
        path = Path(path)

        # Import module-level converter
        from ..serialization import converter

        # Return Data instance from .npz file
        return converter.structure(path, cls)

    def save(self, path: str | Path) -> None:
        """Save the simulation data to a serialized file."""
        # Ensure path is a Path object
        path = Path(path)

        # Create temporary file path for output
        tmp_path = path.with_suffix(path.suffix + ".tmp")

        # Open temporary file for writing
        with open(tmp_path, "wb") as file:
            # Save dictionary representation of data to .npz file
            np.savez(file, **asdict(self))

            # Flush file to ensure all data is written
            file.flush()

            # Force write to disk
            os.fsync(file.fileno())

        # Rename temporary file to final path
        shutil.move(tmp_path, path)


# ---------------------------------
# Monte Carlo Simulation Base Class
# ---------------------------------
@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class Simulation(ABC):
    # Random matrix ensemble
    ensemble: Ensemble = field(converter=Ensemble.create)

    # Number of realizations
    realizs: int = field(
        converter=int,
        validator=gt(0),
        metadata={"dir_name": "realizs", "latex_name": "R"},
    )

    # Simulation data
    data: Data = field(init=False, repr=False, factory=Data)

    @data.validator
    def __data_validator(self, _, value: Any) -> None:
        """Ensure children classes define a Data subclass."""
        # Raise error by default
        raise AttributeError(
            f"Simulation class {type(self).__name__} must define a Data subclass."
        ) from None

    def __attrs_post_init__(self) -> None:
        """Record simulation metadata in Data instance."""
        # Import module-level converter
        from ..serialization import converter

        # Convert simulation instance to dictionary
        sim_dict = converter.unstructure(self)

        # Set simulation metadata in Data instance
        object.__setattr__(self.data, "simulation", sim_dict)

    @classmethod
    def __attrs_init_subclass__(cls) -> None:
        """Register concrete subclasses in the simulation registry."""
        # Include only concrete classes in registry
        if not inspect.isabstract(cls):
            # Convert simulation class to registry key format
            sim_key = re.sub(r"_", "", cls.__name__).lower()

            # Normalize class name to registry key format
            SIMULATION_REGISTRY[sim_key] = cls

    @classmethod
    def create(cls, src: str | Path | dict[str, Any] | Simulation) -> Simulation:
        """Create a simulation instance from a .npz file."""
        # Import module-level converter
        from ..serialization import converter

        # Convert dictionary to simulation instance
        return converter.structure(src, cls)

    @property
    def to_dir(self) -> Path:
        """Generate directory Path used for storing data related to the simulation."""
        # Begin path with class name
        dir_path = Path(self._dir_name)

        # Append ensemble directory representation
        dir_path /= self.ensemble.to_dir

        # Loop through remaining attributes
        for name, attr in fields_dict(type(self)).items():
            # Use only fields labeled for path inclusion
            if attr.metadata.get("dir_name", None) is not None:
                # Sanitize field value for directory representation
                val = re.sub(r"[^\w\-.]", "_", str(asdict(self)[name]))

                # Append string representation of field value to path
                dir_path /= f"{attr.metadata['dir_name']}_{val.replace('.', 'p')}"

        # Return path
        return dir_path

    @property
    def _dir_name(self) -> str:
        """Generate directory name used for storing Simulation instance data."""
        # Insert underscores between words and acronyms in class name
        name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", type(self).__name__)
        return re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name).lower()

    def run(self) -> None:
        """Run the simulation."""
        # Search and store all concrete methods in instance
        inst_methods = inspect.getmembers(self, predicate=inspect.ismethod)

        # Filter and sort methods beginning with "run_part_"
        part_methods = sorted(
            (method for name, method in inst_methods if name.startswith("run_part_")),
            key=lambda m: m.__name__,
        )

        # Execute each part method in order
        for method in part_methods:
            method()
