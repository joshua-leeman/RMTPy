# rmtpy/simulations/_simulation.py

# Postponed evaluation of annotations
from __future__ import annotations

# Standard library imports
import inspect
import re
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
    def __attrs_init_subclass__(cls: type[Data]) -> None:
        """Register concrete subclasses in the data registry."""
        # Include only concrete classes in registry
        if not inspect.isabstract(cls):
            # Convert data class to registry key format
            data_key = re.sub(r"_", "", cls.__name__).lower().replace("data", "")

            # Normalize class name to registry key format
            DATA_REGISTRY[data_key] = cls

    def save(self: Data, path: str | Path) -> None:
        """Save the simulation data to a serialized file."""
        # Save dictionary representation of data to .npz file
        np.savez(path, **asdict(self))


# ---------------------------------
# Monte Carlo Simulation Base Class
# ---------------------------------
@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class Simulation(ABC):
    # Random matrix ensemble
    ensemble: Ensemble = field(converter=Ensemble.create_ensemble)

    # Number of realizations
    realizs: int = field(
        converter=int,
        validator=gt(0),
        metadata={"dir_name": "realizs", "latex_name": "R"},
    )

    # Simulation data
    data: Data = field(init=False, repr=False, factory=Data)

    @data.validator
    def __data_validator(self: Simulation, _, value: Any) -> None:
        """Ensure children classes define a Data subclass."""
        # Raise error by default
        raise AttributeError(
            f"Simulation class {type(self).__name__} must define a Data subclass."
        ) from None

    def __attrs_post_init__(self: Simulation) -> None:
        """Record simulation metadata in Data instance."""
        # Import module-level converter
        from ..serialization import converter

        # Convert simulation instance to dictionary
        sim_dict = converter.unstructure(self)

        # Set simulation metadata in Data instance
        object.__setattr__(self.data, "simulation", sim_dict)

    @classmethod
    def __attrs_init_subclass__(cls: type[Simulation]) -> None:
        """Register concrete subclasses in the simulation registry."""
        # Include only concrete classes in registry
        if not inspect.isabstract(cls):
            # Convert simulation class to registry key format
            sim_key = re.sub(r"_", "", cls.__name__).lower()

            # Normalize class name to registry key format
            SIMULATION_REGISTRY[sim_key] = cls

    @classmethod
    def create_simulation(
        cls: type[Simulation], src: str | Path | dict[str, dict[str, Any]] | Simulation
    ) -> Simulation:
        """Create a simulation instance from a .npz file."""
        # Import module-level converter
        from ..serialization import converter

        # Convert dictionary to simulation instance
        return converter.structure(src, cls)

    @property
    def to_dir(self: Simulation) -> Path:
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
    def _dir_name(self: Simulation) -> str:
        """Generate directory name used for storing Simulation instance data."""
        # Insert underscores between words and acronyms in class name
        name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", type(self).__name__)
        return re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name).lower()
