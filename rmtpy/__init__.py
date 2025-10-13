# rmtpy/__init__.py

# Standard library imports
from typing import Any as __Any

# Local application imports
from .ensembles import Ensemble as __Ensemble, ens_dict as __ens_dict
from .simulations import Simulation as __Simulation, sim_dict as __sim_dict

# Update global namespace with registered objects
globals().update(__ens_dict)
globals().update(__sim_dict)

# Redefine __all__ to include all registered ensembles
__all__ = (
    [ens_name for ens_name in __ens_dict.keys()]
    + [sim_name for sim_name in __sim_dict.keys()]
    + ["ensemble"]
)


# ------------------------------------
# Random Matrix Ensemble Class Factory
# ------------------------------------
def ensemble(**kwargs: __Any) -> __Ensemble:
    """Create a random matrix ensemble instance by name."""
    # Call ensemble constructor with provided keyword arguments as dictionary
    return __Ensemble.create(kwargs)


# ------------------------------------
# Monte Carlo Simulation Class Factory
# ------------------------------------
# def create_simulation(**kwargs: __Any) -> __Simulation:
#     """Get a Monte Carlo simulation by name."""
#     # Call simulation constructor with provided keyword arguments as dictionary
#     return __Simulation.create(kwargs)
