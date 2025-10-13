# rmtpy/ensembles/__init__.py

# Standard library imports
from importlib import import_module
from pathlib import Path

# Local imports
from .base.ensemble import ENSEMBLE_REGISTRY, ensemble, Ensemble
from .base.manybody import ManyBodyEnsemble

# Dynamically import all ensemble modules
path: Path = Path(__file__).parent
for file in path.glob("[!_]*.py"):
    import_module(f".{file.stem}", package=__name__)

# Create dictionary of registered ensembles
ens_dict: dict[str, type[Ensemble]] = {
    ens.__name__: ens for ens in ENSEMBLE_REGISTRY.values()
}

# Update global namespace with registered ensembles
globals().update(ens_dict)

# Redefine __all__ to include all registered ensembles
__all__ = [ens_name for ens_name in ens_dict.keys()]
