# rmtpy/__init__.py

# Standard library imports
from typing import Any as __Any

# Local application imports
from .ensembles import ens_dict as __ens_dict, ensemble
from .simulations import (
    sim_dict as __sim_dict,
    sim_exe_dict as __sim_exe_dict,
    Simulation as __Simulation,
)

# Update global namespace registered ensembles
globals().update(__ens_dict)

# Update global namespace with registered simulations
globals().update(__sim_dict)

# Update global namespace with registered simulation executables
globals().update(__sim_exe_dict)

# Redefine __all__ to include all registered ensembles
__all__ = (
    [ens_name for ens_name in __ens_dict.keys()]
    + [sim_name for sim_name in __sim_dict.keys()]
    + [exe_name for exe_name in __sim_exe_dict.keys()]
    + ["ensemble"]
)
