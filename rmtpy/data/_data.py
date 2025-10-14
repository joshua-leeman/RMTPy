# rmtpy/simulations/base/data.py

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
from attrs import asdict, field, frozen

# Local application imports
from ..ensembles import converter


# -------------------------
# Monte Carlo Data Registry
# -------------------------
DATA_REGISTRY: dict[str, type[Data]] = {}


# ---------------
# Data Base Class
# ---------------
@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class Data(ABC):

    # Metadata
    metadata: dict[str, Any] = field(init=False, factory=dict, repr=False)

    # File name
    file_name: str = field(init=False, repr=False)

    @file_name.default
    def __file_name_default(self) -> str:
        """Generate default filename based on class name."""

        # Convert class name from CamelCase to snake_case
        file_name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", type(self).__name__)
        file_name = file_name.lower()

        # Remove '_data' suffix
        file_name = file_name.replace("_data", "")

        # Return file name
        return file_name

    @classmethod
    def __attrs_init_subclass__(cls) -> None:
        """Register concrete subclasses in the data registry."""

        # Include only concrete classes in registry
        if not inspect.isabstract(cls):
            # Convert data class name from CamelCase to snake_case
            data_key = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", cls.__name__)
            data_key = data_key.lower()

            # Normalize class name to registry key format
            DATA_REGISTRY[data_key] = cls

    @classmethod
    def load(cls, path: str | Path) -> Data:
        """Load simulation data from a serialized file."""

        # Ensure path is a Path object
        path = Path(path)

        # Return Data instance from .npz file
        return converter.structure(path, cls)

    def __attrs_post_init__(self) -> None:
        """Initialize metadata after object creation."""

        # Convert data class name from CamelCase to snake_case
        data_key = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", type(self).__name__)
        data_key = data_key.lower()

        # Add name of data class to metadata
        self.metadata["name"] = data_key

    def save(self, path: str | Path) -> None:
        """Save the simulation data to a serialized file."""

        # Ensure path is a Path object
        path = Path(path)

        # Create temporary file path for output
        tmp_path = path.with_suffix(path.suffix + ".tmp")

        # Open temporary file for writing
        with open(tmp_path, "wb") as file:
            # Save dictionary representation of data to .npz file
            np.savez(file, **asdict(self), allow_pickle=True)

            # Flush file to ensure all data is written
            file.flush()

            # Force write to disk
            os.fsync(file.fileno())

        # Rename temporary file to final path
        shutil.move(tmp_path, path)
