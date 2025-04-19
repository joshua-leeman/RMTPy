# rmtpy.ensembles.__init__.py
"""
This __init__ file sets the environment variables to conduct HPC simpulations for the RMTpy package.
It is grouped into the following sections:
    1. Imports
    2. Ensemble Registry
    3. Functions
"""

# =============================
# 1. Imports
# =============================
# Standard library imports
from pathlib import Path
from importlib import import_module


# =============================
# 2. Ensemble Registry
# =============================
# Instantiate ensemble registry
_ENSEMBLE_REGISTRY = {}

# Loop through all modules in directory
for file in Path(__file__).parent.glob("*.py"):
    # Skip files beginning with an underscore
    if file.name.startswith("_"):
        continue

    # Import module
    module = import_module(f"rmtpy.ensembles.{file.stem}")

    # Get ensemble class
    _ENSEMBLE = getattr(module, module.class_name)

    # Register ensemble class
    _ENSEMBLE_REGISTRY[file.stem] = _ENSEMBLE


# =============================
# 3. Functions
# =============================
# Function to register an ensemble
def get_ensemble(name: str, **kwargs) -> object:
    """
    Get an ensemble from the registry.

    Parameters
    ----------
    name : str
        Name of the ensemble.
    **kwargs : dict
        Additional arguments to pass to the ensemble constructor.

    Returns
    -------
    object
        The ensemble object.
    """
    if name not in _ENSEMBLE_REGISTRY:
        raise ValueError(f"Ensemble '{name}' not found in registry.")

    return _ENSEMBLE_REGISTRY[name](**kwargs)
