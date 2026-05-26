from __future__ import annotations

import attrs
import numpy as np

import rmtpy.density
import rmtpy.validators
from ...data import Data

NUM_TIMES_DEFAULT: int = 6000


def create_array_of_logtimes(form_factors: FormFactorsData) -> None:
    return form_factors.scale * rmtpy.density.array_of_floats(
        support=form_factors.logD_time_support,
        num_pts=form_factors.num_times,
        log_base=form_factors.dimension,
    )


def create_array_of_complex_zeros(form_factors: FormFactorsData) -> None:
    return np.zeros(form_factors.num_times, dtype=np.complex128)


def create_array_of_float_zeros(form_factors: FormFactorsData) -> None:
    return np.zeros(form_factors.num_times, dtype=np.float64)


def finalize_form_factors(form_factors: FormFactorsData) -> None:
    form_factors.compute_form_factors()


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class FormFactorsData(Data):
    dimension: int = attrs.field(
        converter=int,
        validator=attrs.validators.gt(0),
    )
    logD_time_support: tuple[float, float] = attrs.field(
        default=(-1.5, 0.5),
        converter=tuple,
        validator=lambda _, __, support: rmtpy.validators.validate_support(support),
    )
    scale: float = attrs.field(
        default=2 * np.pi,
        converter=float,
        validator=attrs.validators.gt(0.0),
    )
    num_times: int = attrs.field(
        default=NUM_TIMES_DEFAULT,
        converter=int,
        validator=attrs.validators.gt(0),
    )

    times: np.ndarray = attrs.field(
        default=attrs.Factory(create_array_of_logtimes, takes_self=True),
        init=False,
        repr=False,
    )
    first_moment: np.ndarray = attrs.field(
        default=attrs.Factory(create_array_of_complex_zeros, takes_self=True),
        init=False,
        repr=False,
    )
    second_moment: np.ndarray = attrs.field(
        default=attrs.Factory(create_array_of_float_zeros, takes_self=True),
        init=False,
        repr=False,
    )
    form_factor: np.ndarray = attrs.field(
        default=attrs.Factory(create_array_of_float_zeros, takes_self=True),
        init=False,
        repr=False,
    )
    connected_form_factor: np.ndarray = attrs.field(
        default=attrs.Factory(create_array_of_float_zeros, takes_self=True),
        init=False,
        repr=False,
    )

    _realizs_count: int = attrs.field(
        factory=lambda: np.zeros((1,), dtype=np.int64),
        init=False,
        repr=False,
    )

    @property
    def realizs(self) -> int:
        return self._realizs_count[0]

    def compute_moment_contributions(self, levels: np.ndarray) -> None:
        first_moment_contribution: np.ndarray = np.sum(
            np.exp(-1j * np.outer(levels, self.times)), axis=0
        ) / len(levels)
        second_moment_contribution: np.ndarray = np.abs(first_moment_contribution) ** 2

        self.first_moment[:] += first_moment_contribution
        self.second_moment[:] += second_moment_contribution

        self._realizs_count[0] += 1

    def compute_form_factors(self) -> None:
        self.form_factor[:] = self.second_moment / self.realizs
        self.connected_form_factor[:] = (
            self.form_factor - np.abs(self.first_moment / self.realizs) ** 2
        )
