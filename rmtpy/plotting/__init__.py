# rmtpy/plotting/__init__.py

# Standard library imports
from pathlib import Path
from typing import Any

# Third-party imports
import numpy as np
from numpy.lib.npyio import NpzFile

# Local application imports
from ._base import Plot, PLOT_REGISTRY
from ..data import DATA_REGISTRY
from ..ensembles import converter

# Plot from data function
from ._base import plot_data


# ----------------------------
# Register Plot Structure Hook
# ----------------------------
@converter.register_structure_hook
def plot_structure_hook(src: str | Path | dict[str, Any] | NpzFile | Plot, _) -> Plot:
    """Structure hook to convert unstructured data to Plot instance."""

    # If src is already a valid instance, return it
    if isinstance(src, Plot):
        return src

    # If src is a string or Path, load data from npz file
    elif isinstance(src, (str, Path)):
        # Store path for later
        path = Path(src)

        # Load data from .npz file into dictionary
        with np.load(src, allow_pickle=True) as data:
            src_dict = {key: data[key] for key in data.files}

    # If src is not a dictionary, raise TypeError
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

    print("metadata:", metadata)
    print("type(metadata):", type(metadata))

    # Ensure metadata item is a dictionary
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid 'metadata' value in {src}")

    # Retrieve plot name from source
    plot_key = metadata.get("name", None)
    if plot_key is None:
        raise ValueError(f"Missing or invalid 'name' in 'metadata' dict in {src}")

    # Get corresponding Plot class from registry
    if plot_key in PLOT_REGISTRY:
        plot_cls = PLOT_REGISTRY[plot_key]
    else:
        raise ValueError(f"No registered Plot class found in {src}")

    # Get corresponding Data class from registry
    if plot_key in DATA_REGISTRY:
        data_cls = DATA_REGISTRY[plot_key]
    else:
        raise ValueError(f"No registered Data class found for Plot in {src}")

    # Initialize Data instance from source
    data_inst = data_cls.load(path=path)

    # Return constructed Plot instance
    return plot_cls(data=data_inst)
