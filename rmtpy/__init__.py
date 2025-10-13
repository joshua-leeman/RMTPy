# rmtpy/__init__.py

# Local application imports
from .ensembles import rmt_ensemble, ens_dict as __ens_dict
from .simulations import sim_dict as __sim_dict, sim_exe_dict as __sim_exe_dict

# Redefine __all__ to include all registered ensembles
__all__ = (
    [ens_name for ens_name in __ens_dict.keys()]
    + [sim_name for sim_name in __sim_dict.keys()]
    + [exe_name for exe_name in __sim_exe_dict.keys()]
)
