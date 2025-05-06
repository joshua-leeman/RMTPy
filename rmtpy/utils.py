# rmtpy.utils.py


# =======================================
# 1. Imports
# =======================================
# Standard library imports
from __future__ import annotations
import os
import re
from ast import literal_eval
from importlib import import_module
from pathlib import Path
from typing import Union

# Third-party imports
from matplotlib import rcParams


# =======================================
# 2. Dictionary to Ensemble Object
# =======================================
def get_ensemble(ens_args: Union[dict, str]) -> object:
    """Return ensemble object from registry."""
    # Instantiate ensemble registry
    registry = {}

    # Loop through all modules in directory
    for file in Path(__file__).parent.glob("ensembles/*.py"):
        # Skip files beginning with an underscore
        if file.name.startswith("_"):
            continue

        # Import module
        module = import_module(f"rmtpy.ensembles.{file.stem}")

        # Register ensemble class
        registry[file.stem] = getattr(module, module.class_name)

    # If ens_args is a string, convert it to a dictionary
    if isinstance(ens_args, str):
        # Check if string is a valid Python literal
        try:
            ens_args = literal_eval(ens_args)
        except (ValueError, SyntaxError):
            raise ValueError(f"Invalid ensemble string: {ens_args}")

    # Check if ens_args is a dictionary
    if not isinstance(ens_args, dict):
        raise TypeError("Ensemble arguments must be a dictionary or string.")

    # Pop and store name from kwargs
    name = ens_args.pop("name", None).strip().lower()

    # Check if name is provided
    if name is None:
        raise ValueError("Ensemble name must be provided.")

    # Check if ensemble is registered
    if name not in registry:
        raise ValueError(f"Ensemble '{name}' not found in registry.")

    # Return ensemble object
    return registry[name](**ens_args)


# =======================================
# 3. Ensemble from Path
# =======================================
def ensemble_from_path(path: str) -> object:
    """Initializes ensemble from the given path of a data file."""
    # Check if path exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    # Initialize metadata dictionary
    metadata = {}

    # Grabs ensemble name from path
    try:
        ensemble_name = next(datum for datum in path.split("/")[2:])
    except StopIteration:
        raise ValueError(
            f"Invalid path format. Expected format: outputs/<simulation_name>/<ensemble_name>/..."
        )

    # Extract ensemble name and metadata from path
    for part in Path(path).parts:
        # Regex match for metadata key-value pairs
        datum = re.fullmatch(r"(?P<key>\w+)=(?P<val>.+)", part)

        # If regex match is found, store key-value pair in metadata
        if datum is not None:
            # If value contains an underscore, replace it with a dot and convert to float
            if datum.group("val").count("_") == 1:
                metadata[datum.group("key")] = float(
                    datum.group("val").replace("_", ".")
                )
            # Otherwise, convert to string
            else:
                metadata[datum.group("key")] = literal_eval(datum.group("val"))

    # Pop realizations from metadata
    metadata.pop("realizs", None)

    # Add ensemble name to metadata
    metadata["name"] = ensemble_name

    # Return initialized ensemble
    return get_ensemble(metadata)


# =======================================
# 4. Module Configurations
# =======================================
def configure_matplotlib() -> None:
    # Set matplotlib rcParams for plots
    rcParams["axes.axisbelow"] = False
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = "Latin Modern Roman"

    # Try to use LaTeX for rendering
    try:
        rcParams["text.usetex"] = True
        rcParams["text.latex.preamble"] = "\n".join(
            [
                r"\usepackage{amsmath}",
                r"\newcommand{\ensavg}[1]{\langle\hspace{-0.7ex}\langle #1 \rangle\hspace{-0.7ex}\rangle}",
            ]
        )
    except:
        pass
