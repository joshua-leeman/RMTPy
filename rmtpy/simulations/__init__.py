# rmtpy/simulations/__init__.py

# Standard library imports
import json
import re
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any
from types import FunctionType

# Local application imports
from ._simulation import (
    Simulation,
    SIMULATION_REGISTRY,
    SIMULATION_EXECUTABLE_REGISTRY,
)
from .spectral_simulation import spectral_statistics
from ..data import DATA_REGISTRY
from ..ensembles import normalize_dict, converter

# Get directory that contains this file
path: Path = Path(__file__).parent

# Dynamically import all simulation modules
path: Path = Path(__file__).parent
for file in path.glob("[!_]*.py"):
    import_module(f".{file.stem}", package=__name__)

# Create dictionary of registered simulations
sim_dict: dict[str, type[Simulation]] = {
    sim.__name__: sim for sim in SIMULATION_REGISTRY.values()
}

# Create dictionary of registered simulation executables
sim_exe_dict: dict[str, type[FunctionType]] = {
    exe.__name__: exe for exe in SIMULATION_EXECUTABLE_REGISTRY.values()
}

# Redefine __all__ to include all registered simulations
__all__ = [sim_name for sim_name in sim_dict.keys()] + [
    exe_name for exe_name in sim_exe_dict.keys()
]


# ---------------------------------
# Default Structure Hook Dictionary
# ---------------------------------
SIM_STRUCTURE_HOOKS: dict[str, Callable] = {
    key: converter.get_structure_hook(val) for key, val in SIMULATION_REGISTRY.items()
}


# -----------------------------------
# Default Unstructure Hook Dictionary
# -----------------------------------
SIM_UNSTRUCTURE_HOOKS: dict[str, Callable] = {
    key: converter.get_unstructure_hook(val) for key, val in SIMULATION_REGISTRY.items()
}


# ----------------------------------
# Register Simulation Structure Hook
# ----------------------------------
@converter.register_structure_hook
def sim_structure_hook(src: str | Path | dict[str, Any] | Simulation, _) -> Simulation:
    """Convert serialized data from a directory to a Simulation instance."""

    # If src is already a valid instance, return it
    if type(src) in SIMULATION_REGISTRY.values():
        return src

    # Check if src is a path-like object
    elif isinstance(src, (str, Path)):
        # Ensure src is a Path object
        path = Path(src)

        # Load simulation metadata from .json file
        with open(path / "metadata.json", "r") as file:
            metadata = json.load(file)

    # Check if src is a dictionary
    elif isinstance(src, dict):
        metadata = src

    # If src is of invalid type, raise TypeError
    else:
        raise TypeError(
            f"Expected str, Path, dict, or Simulation, got {type(src).__name__}"
        )

    # Normalize metadata dictionary
    sim_dict = normalize_dict(metadata, SIMULATION_REGISTRY)

    # Extract simulation name from normalized dictionary, raise error if invalid
    sim_name = sim_dict.pop("name")
    if not isinstance(sim_name, str):
        raise ValueError(f"Invalid simulation name type: {type(sim_name).__name__}")

    # Convert simulation name to registry key format
    sim_key = re.sub(r"_", "", sim_name).lower()

    # Retrieve simulation class from registry
    sim_cls = SIMULATION_REGISTRY[sim_key]

    # Extract simulation arguments from normalized dictionary, raise error if invalid
    sim_args = sim_dict.pop("args")
    if not isinstance(sim_args, dict):
        raise ValueError(f"Invalid simulation args type: {type(sim_args).__name__}")

    # Use base structure hook to convert normalized dictionary
    sim_inst: Simulation = SIM_STRUCTURE_HOOKS[sim_key](sim_args, sim_cls)

    # Load associated data from subdirectories if path is a directory
    if isinstance(src, (str, Path)):
        # Create generator of subdirectories in simulation directory
        data_dirs = (folder for folder in path.iterdir() if folder.is_dir())

        # Loop through folders in simulation directory
        for folder in data_dirs:
            # Get data class type from folder name
            data_cls = DATA_REGISTRY.get(folder.name, None)
            if data_cls is None:
                continue

            # Load data from .npz file
            data = data_cls.load(folder / f"{folder.name}.npz")

            # Set data attribute on simulation instance
            object.__setattr__(sim_inst, folder.name + "_data", data)

    # Return loaded simulation instance
    return sim_inst
