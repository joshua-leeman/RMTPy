# rmtpy/plotting/__init__.py

# Standard library imports
from pathlib import Path
from typing import Any

# Third-party imports
import numpy as np
from numpy.lib.npyio import NpzFile

# Local application imports
from ._base import Plot, PLOT_REGISTRY
from ..data import normalize_metadata, normalize_source, DATA_REGISTRY
from ..ensembles import converter

# Plot from data function
from ._base import plot_data


# ----------------------------
# Register Plot Structure Hook
# ----------------------------
@converter.register_structure_hook
def plot_structure_hook(src: str | Path | dict[str, Any] | NpzFile | Plot, _) -> Plot:
    """Structure hook to convert unstructured data to Plot instance."""

    # Convert source to normalized dictionary
    src_dict = normalize_source(src)

    # Normalize metadata dictionary
    metadata = normalize_metadata(src_dict["metadata"])

    # Replace metadata in source dictionary with normalized version
    src_dict["metadata"] = metadata

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
    data_inst = converter.structure(src_dict, data_cls)

    # Return constructed Plot instance
    return plot_cls(data=data_inst)
