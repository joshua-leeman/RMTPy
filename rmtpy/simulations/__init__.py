# rmtpy/simulations/__init__.py

# Standard library imports
from importlib import import_module
from pathlib import Path
from types import FunctionType

# Local application imports
from ._simulation import (
    Simulation,
    SIMULATION_REGISTRY,
    SIMULATION_EXECUTABLE_REGISTRY,
)
from .spectral_simulation import spectral_statistics

# Get directory that contains this file
path: Path = Path(__file__).parent

# Dynamically import all Python modules in this directory
for subdir in path.iterdir():
    if subdir.is_dir() and (subdir / "__init__.py").exists():
        import_module(f".{subdir.name}", package=__name__)

# Create dictionary of registered simulations
sim_dict: dict[str, type[Simulation]] = {
    sim.__name__: sim for sim in SIMULATION_REGISTRY.values()
}

# Create dictionary of registered simulation executables
sim_exe_dict: dict[str, type[FunctionType]] = {
    exe.__name__: exe for exe in SIMULATION_EXECUTABLE_REGISTRY.values()
}

# Redefine __all__ to include all registered simulations
__all__ = [sim_name for sim_name in sim_dict.keys()] + [
    exe_name for exe_name in sim_exe_dict.keys()
]
