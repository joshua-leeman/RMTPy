# rmtpy/data/__init__.py

# Standard library imports
from pathlib import Path
from typing import Any

# Third-party imports
import numpy as np
from numpy.lib.npyio import NpzFile

# Local application imports
from ._data import Data, DATA_REGISTRY
from ..ensembles import converter


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
        # Load data from .npz file into dictionary
        with np.load(src, allow_pickle=True) as data:
            src_dict = {key: data[key] for key in data.files}

    # If src is not a dictionary or npz file, raise TypeError
    elif not isinstance(src, dict):
        raise TypeError(
            f"Expected path, dict, npz file, or Data, got {type(src).__name__}"
        )

    # Retrieve data metadata from source
    metadata = src_dict.get("metadata", None)
    if metadata is None:
        raise ValueError(f"Missing 'metadata' in {src}")
    else:
        metadata = metadata.item()
        src_dict["metadata"] = metadata

    # Ensure metadata item is a dictionary
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid 'metadata' value in {src}")

    # Retrieve data name from source
    data_key = metadata.get("name", None)
    if data_key is None:
        raise ValueError(f"Missing or invalid 'name' in 'metadata' dict in {src}")

    # Get corresponding Data class from registry
    if data_key in DATA_REGISTRY:
        data_cls = DATA_REGISTRY[data_key]
    else:
        raise ValueError(f"No registered Data class found in {src}")

    # Initialize default Data instance
    data_inst = data_cls()

    print("data_inst.metadata before:", data_inst.metadata)

    # Update data instance with source attributes
    for key in src_dict:
        object.__setattr__(data_inst, key, src_dict[key])

    # Return constructed Data instance
    return data_inst
