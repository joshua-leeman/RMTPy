# rmtpy/data/evolve_cdo_data/cdo_dynamics_data.py

# Postponed evaluation of annotations
from __future__ import annotations

# Third-party imports
import numpy as np
from attrs import frozen, field
from attrs.validators import gt

# Local application imports
from .._data import Data


# -----------------------
# CDO Dynamics Data Class
# -----------------------
@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class CDODynamicsData(Data):

    # Number of times for CDO dynamics
    num_times: int = field(converter=int, validator=gt(0), default=1000)

    # Hilbert space dimension
    dim: int = field(converter=int, validator=gt(0))

    # Logarithmic range for times
    logt_i: float = -1.0  # base = dim
    logt_f: float = 1.5  # base = dim

    # Times for CDO dynamics
    times: np.ndarray = field(init=False, repr=False)

    # Probabilities data
    probs: np.ndarray = field(init=False, repr=False)

    # Classical purity data
    c_purity: np.ndarray = field(init=False, repr=False)

    # Quantum purity data
    q_purity: np.ndarray = field(init=False, repr=False)

    # von Neumann entropy data
    entropy: np.ndarray = field(init=False, repr=False)

    # Kullback-Leibler divergence data
    kl_div: np.ndarray = field(init=False, repr=False)

    @times.default
    def __times_default(self) -> np.ndarray:
        """Initialize the times array."""

        # Create logarithmically spaced times
        times = np.logspace(self.logt_i, self.logt_f, self.num_times - 1)

        # Add time zero at the beginning and return
        return np.insert(times, 0, 0.0)

    @probs.default
    def __probs_default(self) -> np.ndarray:
        """Initialize the probabilities array."""

        # Return empty array with shape T x D
        return np.empty((self.num_times, self.dim), dtype=float, order="F")

    @c_purity.default
    def __c_purity_default(self) -> np.ndarray:
        """Initialize the classical purity array."""

        # Return empty array with shape T
        return np.empty(self.num_times, dtype=float, order="F")

    @q_purity.default
    def __q_purity_default(self) -> np.ndarray:
        """Initialize the quantum purity array."""

        # Return empty array with shape T
        return np.empty(self.num_times, dtype=float, order="F")

    @entropy.default
    def __entropy_default(self) -> np.ndarray:
        """Initialize the von Neumann entropy array."""

        # Return empty array with shape T
        return np.empty(self.num_times, dtype=float, order="F")

    @kl_div.default
    def __kl_div_default(self) -> np.ndarray:
        """Initialize the Kullback-Leibler divergence array."""

        # Return empty array with shape T
        return np.empty(self.num_times, dtype=float, order="F")
