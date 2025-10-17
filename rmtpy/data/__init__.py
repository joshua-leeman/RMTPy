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

# Load Data Function
from ._data import load_data

# Import concrete Data classes to register them
from .spectral_statistics_data import (
    FormFactorsData,
    SpectralDensityData,
    SpacingHistogramData,
)


# ---------------------------
# Normalize Metadata Function
# ---------------------------
def normalize_metadata(metadata: Any) -> dict[str, Any]:
    """Normalize metadata input to a dictionary for Data initialization."""

    # Unwrap metadata if it is a 0-dim numpy array
    if isinstance(metadata, np.ndarray) and metadata.dtype == object:
        metadata = metadata.item()

    # If metadata is a dictionary, return it
    if isinstance(metadata, dict):
        return metadata
    else:
        raise TypeError(f"Expected dict, got {type(metadata).__name__}")


# -------------------------
# Normalize Source Function
# -------------------------
def normalize_source(src: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Normalize source input to a dictionary for Data initialization."""

    # If src is a Path-like, load data from npz file into dictionary
    if isinstance(src, (str, Path)):
        with np.load(src, allow_pickle=True) as data:
            src_dict = {key: data[key] for key in data.files}

    # If src is already a dictionary, use it directly
    elif isinstance(src, dict):
        src_dict = src

    # If src is of invalid type, raise TypeError
    else:
        raise TypeError(f"Expected path, dict, npz file, got {type(src).__name__}")

    # Return normalized dictionary
    return src_dict


# ----------------------------
# Register Data Structure Hook
# ----------------------------
@converter.register_structure_hook
def data_structure_hook(src: str | Path | dict[str, Any] | NpzFile | Data, _) -> Data:
    """Structure hook to convert unstructured data to Data instance."""

    # Convert source to normalized dictionary
    src_dict = normalize_source(src)

    # Normalize metadata dictionary
    metadata = normalize_metadata(src_dict["metadata"])

    # Replace metadata in source dictionary with normalized version
    src_dict["metadata"] = metadata

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

    # Update data instance with source attributes
    for key in src_dict:
        object.__setattr__(data_inst, key, src_dict[key])

    # Return constructed Data instance
    return data_inst
