# rmtpy/simulations/__init__.py

# Standard library imports
from importlib import import_module
from pathlib import Path

# Local imports
from ._simulation import Data, Simulation, DATA_REGISTRY, SIMULATION_REGISTRY

# Dynamically import all simulation modules
path: Path = Path(__file__).parent
for file in path.glob("[!_]*.py"):
    import_module(f".{file.stem}", package=__name__)

# Create dictionary of registered simulations
sim_dict: dict[str, type[Simulation]] = {
    sim.__name__: sim for sim in SIMULATION_REGISTRY.values()
}

# Update global namespace with registered simulations
globals().update(sim_dict)

# Redefine __all__ to include all registered simulations
__all__ = [sim_name for sim_name in sim_dict.keys()]
