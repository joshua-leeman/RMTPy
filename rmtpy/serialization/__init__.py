# rmtpy/serialization/__init__.py

# Standard library imports
import re
from importlib import import_module
from pathlib import Path
from typing import Any

# Third-party imports
import numpy as np
from attrs import fields_dict
from cattrs import Converter


# ----------------------
# Module-level Converter
# ----------------------
# Initialize converter instance
converter = Converter()

# Unstructure hook: from np.dtype to JSON-serializable type
converter.register_unstructure_hook(np.dtype, lambda dt: dt.name)

# Structure hook: from string to np.dtype
converter.register_structure_hook(np.dtype, lambda s, _: np.dtype(s))


# ---------------------------------
# Normalize Unstructured Dictionary
# ---------------------------------
def normalize_dict(
    src: dict[str, Any], registry: dict[str, type]
) -> dict[str, str | dict[str, Any]]:
    """Normalize form of class dictionary."""
    # If src is not a dictionary, raise TypeError
    if not isinstance(src, dict):
        raise TypeError(f"Expected a dictionary, got {type(src).__name__}")

    # Initialize empty dictionary
    norm_dict = {}

    # Loop through values in input dictionary
    for val in src.values():
        # Only consider string values
        if isinstance(val, str):
            # Normalize value to registry key format
            key = re.sub(r"[_ ]", "", val).lower()

            # Check if key is in class registry
            if key in registry:
                # Retrieve class class from registry
                regd_cls = registry[key]

                # Store class name in norm_dict and break loop
                norm_dict["name"] = regd_cls.__name__
                break

    # If norm_dict is still empty, raise KeyError
    if not norm_dict:
        raise KeyError("Registered class name not found in dictionary")

    # Determine set of expected arguments for registered class
    exp_args = {arg for arg, attr in fields_dict(regd_cls).items() if attr.init}

    # Check if there are no expected arguments
    if not exp_args:
        # Update norm_dict with empty args
        norm_dict.update({"args": {}})

        # Return normalized dictionary
        return norm_dict

    # Loop through values in input dictionary again
    for val in src.values():
        # Check if value is a dictionary with expected arguments
        if isinstance(val, dict) and set(val.keys()).issubset(exp_args):
            # Update norm_dict with expected arguments and break loop
            norm_dict.update({"args": val})

            # Return normalized dictionary
            return norm_dict

    # Else, check if expected arguments are top-level keys
    if set(src.keys()).isdisjoint(exp_args):
        raise TypeError(f"Invalid or missing arguments for {regd_cls.__name__}")

    # Extract top-level expected arguments and update norm_dict
    norm_dict.update({"args": {arg: src[arg] for arg in src if arg in exp_args}})

    # Return normalized dictionary
    return norm_dict


# --------------------------
# Register Custom Converters
# --------------------------
# Dynamically import all converter modules
path: Path = Path(__file__).parent
for file in path.glob("[!_]*.py"):
    import_module(f".{file.stem}", package=__name__)
