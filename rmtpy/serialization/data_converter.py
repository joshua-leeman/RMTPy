# rmtpy/serialization/data_converter.py

# Standard library imports
import re
from pathlib import Path
from typing import Any

# Third-party imports
import numpy as np
from numpy.lib.npyio import NpzFile

# Local application imports
from ..simulations import Data, DATA_REGISTRY
from . import converter


# ----------------------------
# Register Data Structure Hook
# ----------------------------
@converter.register_structure_hook
def data_structure_hook(src: str | Path | dict[str, Any] | NpzFile | Data, _) -> Data:
    """Structure hook to convert unstructured data to Data instance."""

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
    metadata = src.get("metadata", None)
    if not isinstance(metadata, dict):
        raise ValueError(f"Missing or invalid 'metadata' key in {src}")

    # Retrieve simulation name from source
    data_name = metadata.get("name", None)
    if not isinstance(data_name, str):
        raise ValueError(f"Missing or invalid 'name' in 'metadata' dict in {src}")

    # Convert simulation name to registry key format
    data_key = re.sub(r"_", "", data_name).lower().replace("data", "")

    # Get corresponding data class from registry
    if data_key in DATA_REGISTRY:
        data_cls = DATA_REGISTRY[data_key]
    else:
        raise ValueError(f"No registered data class found in {src}")

    # Initialize default Data instance
    data_inst = data_cls()

    # Update data instance with simulation data
    for key, val in src.items():
        object.__setattr__(data_inst, key, val)

    # Return constructed Data instance
    return data_inst
