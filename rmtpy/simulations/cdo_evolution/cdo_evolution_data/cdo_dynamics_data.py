from __future__ import annotations

import numpy as np
from attrs import frozen, field
from attrs.validators import gt

from ..._data import Data


@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class CDODynamicsData(Data):

    num_times: int = field(converter=int, validator=gt(0), default=1000)

    dim: int = field(converter=int, validator=gt(0))

    logt_i: float = -1.0
    logt_f: float = 1.5

    times: np.ndarray = field(init=False, repr=False)

    probs: np.ndarray = field(init=False, repr=False)

    c_purity: np.ndarray = field(init=False, repr=False)

    q_purity: np.ndarray = field(init=False, repr=False)

    entropy: np.ndarray = field(init=False, repr=False)

    kl_div: np.ndarray = field(init=False, repr=False)

    @times.default
    def __times_default(self) -> np.ndarray:
        """Initialize the times array."""

        times = np.logspace(self.logt_i, self.logt_f, self.num_times - 1)

        return np.insert(times, 0, 0.0)

    @probs.default
    def __probs_default(self) -> np.ndarray:
        """Initialize the probabilities array."""

        return np.empty((self.num_times, self.dim), dtype=float, order="F")

    @c_purity.default
    def __c_purity_default(self) -> np.ndarray:
        """Initialize the classical purity array."""

        return np.empty(self.num_times, dtype=float, order="F")

    @q_purity.default
    def __q_purity_default(self) -> np.ndarray:
        """Initialize the quantum purity array."""

        return np.empty(self.num_times, dtype=float, order="F")

    @entropy.default
    def __entropy_default(self) -> np.ndarray:
        """Initialize the von Neumann entropy array."""

        return np.empty(self.num_times, dtype=float, order="F")

    @kl_div.default
    def __kl_div_default(self) -> np.ndarray:
        """Initialize the Kullback-Leibler divergence array."""

        return np.empty(self.num_times, dtype=float, order="F")
