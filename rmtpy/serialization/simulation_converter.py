# rmtpy/serialization/simulation_converter.py

# Standard library imports
import re
from pathlib import Path
from collections.abc import Callable
from typing import Any

# Third-party imports
import numpy as np
from numpy.lib.npyio import NpzFile

# Local imports
from ..ensembles import ENSEMBLE_REGISTRY
from ..simulations import Data, Simulation, DATA_REGISTRY, SIMULATION_REGISTRY
from ._converter import normalize_dict, converter


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


# ----------------------------
# Register Data Structure Hook
# ----------------------------
@converter.register_structure_hook
def data_structure_hook(src: str | Path | dict[str, Any] | NpzFile | Data, _) -> Data:
    # If src is already a valid instance, return it
    if isinstance(src, Data):
        return src

    # If src is a string or Path, load data from npz file
    elif isinstance(src, (str, Path)):
        # Load data from .npz file
        src = np.load(src, allow_pickle=True)

    # If src is not a dictionary or npz file, raise TypeError
    elif not isinstance(src, (dict, NpzFile)):
        raise TypeError(
            f"Expected path, dict, npz file, or Data, got {type(src).__name__}"
        )

    # Determine simulation name from source
    simulation = src.get("simulation", None)
    if not isinstance(simulation, dict):
        raise ValueError(f"Missing or invalid 'simulation' key in {src}")

    # Retrieve simulation name from source
    sim_name = simulation.get("name", None)
    if not isinstance(sim_name, str):
        raise ValueError(f"Missing or invalid 'name' in 'simulation' dict in {src}")

    # Convert simulation name to registry key format
    sim_key = re.sub(r"_", "", sim_name).lower()

    # Get corresponding data class from registry
    if sim_key in DATA_REGISTRY:
        data_cls = DATA_REGISTRY[sim_key]
    else:
        raise ValueError(f"No registered data class found in {src}")

    # Initialize default Data instance
    data_inst = data_cls()

    # Update data instance with simulation data
    for key, val in src.items():
        object.__setattr__(data_inst, key, val)

    # Return constructed Data instance
    return data_inst


# ----------------------------------
# Register Simulation Structure Hook
# ----------------------------------
@converter.register_structure_hook
def sim_structure_hook(src: str | Path | dict[str, Any] | Simulation, _) -> Simulation:
    """Convert a general dictionary to a Simulation instance."""
    # If src is already a valid instance, return it
    if type(src) in SIMULATION_REGISTRY.values():
        return src

    # If src is a string or Path, load simulation data from npz file
    elif isinstance(src, (str, Path)):
        # Load data from .npz file
        data = converter.structure(src, Data)

        # Extract simulation dictionary from data
        sim_dict = data.simulation

        # Normalize simulation dictionary
        sim_dict = normalize_dict(sim_dict, SIMULATION_REGISTRY)

    # Else, try building unstructured Ensemble instance
    else:
        # Set data to None by default
        data = None

        # Normalize source dictionary
        try:
            ens_dict = {"ensemble": normalize_dict(src, ENSEMBLE_REGISTRY)}
        except KeyError:
            ens_dict = {}

        # Create copy of source dictionary
        src = src.copy()

        # Update new source dictionary with ensemble dictionary
        src.update(ens_dict)

        # Normalize source dictionary
        sim_dict = normalize_dict(src, SIMULATION_REGISTRY)

    # Extract simulation name from normalized dictionary, raise error if invalid
    sim_name = sim_dict.pop("name")
    if not isinstance(sim_name, str):
        raise ValueError(f"Invalid simulation name type: {type(sim_name).__name__}")

    # Convert simulation name to registry key format
    sim_key = re.sub(r"_", "", sim_name).lower()

    # Get corresponding simulation class from registry
    sim_cls = SIMULATION_REGISTRY[sim_key]

    # Extract simulation arguments from normalized dictionary, raise error if invalid
    sim_args = sim_dict.pop("args")
    if not isinstance(sim_args, dict):
        raise ValueError(f"Invalid simulation args type: {type(sim_args).__name__}")

    # Use base structure hook to convert normalized dictionary to instance
    sim_inst: Simulation = SIM_STRUCTURE_HOOKS[sim_key](sim_args, sim_cls)

    # Check if data is provided
    if data is not None:
        # Set simulation data
        object.__setattr__(sim_inst, "data", data)

    # Return constructed Simulation instance
    return sim_inst


# ------------------------------------
# Register Simulation Unstructure Hook
# ------------------------------------
@converter.register_unstructure_hook
def sim_unstructure_hook(sim: Simulation) -> dict[str, str | dict[str, Any]]:
    """Convert an Simulation instance to a normalized dictionary."""
    # Initialize empty normalized dictionary
    sim_dict = {}

    # Store simulation name in normalized dictionary
    sim_dict["name"] = type(sim).__name__

    # Convert simulation name to registry key format
    sim_key = re.sub(r"_", "", sim_dict["name"]).lower()

    # Use default unstructure hook to get arguments
    sim_args = SIM_UNSTRUCTURE_HOOKS[sim_key](sim)

    # Store simulation arguments in normalized dictionary
    sim_dict["args"] = sim_args

    # Return normalized dictionary
    return sim_dict
