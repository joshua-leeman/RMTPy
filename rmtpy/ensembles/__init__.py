# rmtpy/ensembles/__init__.py

# Standard library imports
import re
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any

# Local application imports
from ._base import (
    Ensemble,
    GaussianEnsemble,
    ManyBodyEnsemble,
    normalize_dict,
    rmt_ensemble,
    converter,
    ENSEMBLE_REGISTRY,
)

# Dynamically import all ensemble modules
path: Path = Path(__file__).parent
for file in path.glob("[!_]*.py"):
    import_module(f".{file.stem}", package=__name__)

# Create dictionary of registered ensembles
ens_dict: dict[str, type[Ensemble]] = {
    ens.__name__: ens for ens in ENSEMBLE_REGISTRY.values()
}

# Update global namespace with registered ensembles
globals().update(ens_dict)

# Redefine __all__ to include all registered ensembles
__all__ = [ens_name for ens_name in ens_dict.keys()]


# ---------------------------------
# Default Structure Hook Dictionary
# ---------------------------------
ENS_STRUCTURE_HOOKS: dict[str, Callable] = {
    key: converter.get_structure_hook(val) for key, val in ENSEMBLE_REGISTRY.items()
}


# -----------------------------------
# Default Unstructure Hook Dictionary
# -----------------------------------
ENS_UNSTRUCTURE_HOOKS: dict[str, Callable] = {
    key: converter.get_unstructure_hook(val) for key, val in ENSEMBLE_REGISTRY.items()
}


# --------------------------------
# Register Ensemble Structure Hook
# --------------------------------
@converter.register_structure_hook
def ens_structure_hook(src: dict[str, Any] | Ensemble, _) -> Ensemble:
    """Convert a general dictionary to an Ensemble instance."""

    # If src is already a valid instance, return it
    if type(src) in ENSEMBLE_REGISTRY.values():
        return src

    # Normalize input dictionary
    ens_dict = normalize_dict(src, ENSEMBLE_REGISTRY)

    # Extract ensemble name from normalized dictionary, raise error if invalid
    ens_name = ens_dict.pop("name")
    if not isinstance(ens_name, str):
        raise ValueError(f"Invalid ensemble name type: {type(ens_name).__name__}")

    # Convert ensemble name to registry key format
    ens_key = re.sub(r"_", "", ens_name).lower()

    # Retrieve ensemble class from registry
    ens_cls = ENSEMBLE_REGISTRY[ens_key]

    # Extract ensemble arguments from normalized dictionary, raise error if invalid
    ens_args = ens_dict.pop("args")
    if not isinstance(ens_args, dict):
        raise ValueError(f"Invalid ensemble args type: {type(ens_args).__name__}")

    # Use base structure hook to convert normalized dictionary
    ens_inst: Ensemble = ENS_STRUCTURE_HOOKS[ens_key](ens_args, ens_cls)

    # Set random number generator state
    ens_inst.set_rng_state(src.get("rng_state", None))

    # Return constructed Ensemble instance
    return ens_inst


# ----------------------------------
# Register Ensemble Unstructure Hook
# ----------------------------------
@converter.register_unstructure_hook
def ens_unstructure_hook(ensemble: Ensemble) -> dict[str, str | dict[str, Any]]:
    """Convert an Ensemble instance to a normalized dictionary."""

    # Initialize empty normalized dictionary
    ens_dict = {}

    # Store ensemble name in normalized dictionary
    ens_dict["name"] = type(ensemble).__name__

    # Convert ensemble name to registry key format
    ens_key = re.sub(r"_", "", ens_dict["name"]).lower()

    # Use default unstructure hook to get arguments
    ens_args = ENS_UNSTRUCTURE_HOOKS[ens_key](ensemble)

    # Store ensemble arguments in normalized dictionary
    ens_dict["args"] = ens_args

    # Store state of ensemble's random number generator
    ens_dict["rng_state"] = ensemble.rng_state

    # Return normalized dictionary
    return ens_dict
