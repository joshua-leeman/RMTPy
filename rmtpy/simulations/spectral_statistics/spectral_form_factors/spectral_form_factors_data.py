from __future__ import annotations

import numpy as np
from attrs import frozen, field
from attrs.validators import gt

from ..._data import Data


@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class FormFactorsData(Data):
    dimension: int = field()
    log_D_time_support: tuple[float, float] = field(
        default=(-1.5, 0.5)
    )  # log scale base dim
    scale: float = field(default=2 * np.pi)
    num_times: int = field(converter=int, validator=gt(0), default=3000)

    times: np.ndarray = field(init=False, repr=False)

    @times.default
    def _default_times(self) -> np.ndarray:
        log_times = np.logspace(
            self.log_D_time_support[0], self.log_D_time_support[1], self.num_times
        )
        log_times **= np.log(self.dimension) / np.log(10)
        return self.scale * log_times

    first_moment: np.ndarray = field(init=False, repr=False)

    @first_moment.default
    def _default_first_moment(self) -> np.ndarray:
        return np.zeros(self.num_times, dtype=np.complex128)

    second_moment: np.ndarray = field(init=False, repr=False)

    @second_moment.default
    def _default_second_moment(self) -> np.ndarray:
        return np.zeros(self.num_times, dtype=np.float64)

    form_factor: np.ndarray = field(init=False, repr=False)

    @form_factor.default
    def _default_form_factor(self) -> np.ndarray:
        return np.empty(self.num_times, dtype=np.float64)

    connected_form_factor: np.ndarray = field(init=False, repr=False)

    @connected_form_factor.default
    def _default_connected_form_factor(self) -> np.ndarray:
        return np.empty(self.num_times, dtype=np.float64)

    _realizs_count: int = field(
        init=False, repr=False, factory=lambda: np.zeros((1,), dtype=int)
    )

    @property
    def realizs(self) -> int:
        return self._realizs_count[0]

    def compute_moment_contributions(self, levels: np.ndarray) -> None:
        first_moment_contribution = np.sum(
            np.exp(-1j * np.outer(levels, self.times)), axis=0
        ) / len(levels)
        self.first_moment[:] += first_moment_contribution

        second_moment_contribution = np.abs(first_moment_contribution) ** 2
        self.second_moment[:] += second_moment_contribution

        self._realizs_count[0] += 1

    def compute_form_factors(self) -> None:
        self.form_factor[:] = self.second_moment / self.realizs
        self.connected_form_factor[:] = (
            self.form_factor - np.abs(self.first_moment / self.realizs) ** 2
        )
