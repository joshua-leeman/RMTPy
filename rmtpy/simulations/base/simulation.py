# rmtpy/simulations/base/simulation.py

# Postponed evaluation of annotations
from __future__ import annotations

# Standard library imports
import inspect
import json
import re
from abc import ABC
from pathlib import Path
from typing import Any

# Third-party imports
from attrs import asdict, field, fields_dict, frozen
from attrs.validators import gt

# Local imports
from .data import Data
from ...ensembles import Ensemble
from ...plotting.base.plot import Plot


# -------------------------------
# Monte Carlo Simulation Registry
# -------------------------------
SIMULATION_REGISTRY: dict[str, type[Simulation]] = {}


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

    # Metadata
    metadata: dict[str, Any] = field(init=False, factory=dict, repr=False)

    # Directory name
    dir_name: str = field(init=False, repr=False)

    @dir_name.default
    def __dir_name_default(self) -> str:
        """Generate default directory name based on class name."""
        # Insert underscores between words and acronyms in class name
        name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", type(self).__name__)
        return re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name).lower()

    @ensemble.validator
    def __ensemble_is_concrete(self, _, value: Ensemble) -> None:
        """Ensure ensemble is a concrete subclass of Ensemble."""
        if inspect.isabstract(value):
            raise ValueError(
                f"Ensemble must be a concrete subclass of Ensemble, got {type(value).__name__} instead."
            )

    @classmethod
    def __attrs_init_subclass__(cls) -> None:
        """Register concrete subclasses in the simulation registry."""
        # Include only concrete classes in registry
        if not inspect.isabstract(cls):
            # Convert simulation class to registry key format
            sim_key = re.sub(r"_", "", cls.__name__).lower()

            # Normalize class name to registry key format
            SIMULATION_REGISTRY[sim_key] = cls

    def __attrs_post_init__(self) -> None:
        """Initialize metadata after object creation."""
        # Import module-level converter
        from ...serialization import converter

        # Add simulation name to metadata
        self.metadata["name"] = type(self).__name__

        # Initialize arguments dictionary
        self.metadata["args"] = {}

        # Add ensemble name to metadata
        self.metadata["args"]["ensemble"] = converter.unstructure(self.ensemble)

        # Add number of realizations to metadata
        self.metadata["args"]["realizs"] = self.realizs

        # Create generator of data attributes
        data_attrs = (attr for attr in asdict(self).values() if isinstance(attr, Data))

        # Loop through data objects
        for data in data_attrs:
            # Store simulation metadata to data object
            data.metadata.update({"simulation": self.metadata.copy()})

    def save_data(self, out_dir: str | Path) -> None:
        """Save simulation results to disk."""
        # Save metadata to .json file
        with open(out_dir / "metadata.json", "w") as file:
            json.dump(self.metadata, file, indent=4)

        # Create generator of data attributes
        data_attrs = (attr for attr in asdict(self).values() if isinstance(attr, Data))

        # Loop through data objects
        for data in data_attrs:
            # Generate output path for data object
            out_path = out_dir / data.file_name / f"{data.file_name}.npz"

            # Make sure directories exist
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Save data to disk
            data.save(out_path)

    def save_plots(self, out_dir: str | Path) -> None:
        """Save simulation plots to disk."""
        # Create generator of plot attributes
        plot_attrs = (attr for attr in asdict(self).values() if isinstance(attr, Plot))

        # Loop through plot objects
        for plot in plot_attrs:
            # Generate output path for plot object
            out_path = out_dir / plot.file_name / f"{plot.file_name}.png"

            # Make sure directories exist
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Save plot to disk
            plot.plot(path=out_path)

    @property
    def to_dir(self) -> Path:
        """Generate directory Path used for storing simulation data."""
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
        """Generate directory name used for storing simulation data."""
        # Insert underscores between words and acronyms in class name
        name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", type(self).__name__)
        return re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name).lower()
