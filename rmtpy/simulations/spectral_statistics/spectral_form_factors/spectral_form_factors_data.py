from typing import ClassVar, cast

import attrs
import numpy as np
from scipy.special import jn_zeros

from ....density import Support, array_of_floats
from ....ensembles import ManyBodyEnsemble
from ....validators import is_valid_support, to_support_pair
from ...base_data import Data
from ...statistics import LOG_D_TIME_SUPPORT, LOG_D_UNFOLDED_TIME_SUPPORT

NUM_TIMES: int = 6000

TIME_CHUNK_SIZE: int = 1024


def _build_complex_zeros(
    form_factors: FormFactorsData,
) -> np.ndarray[tuple[int], np.dtype[np.complexfloating]]:
    return np.zeros(form_factors.num_times, dtype=np.complex128)


def _build_float_zeros(
    form_factors: FormFactorsData,
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    return np.zeros(form_factors.num_times, dtype=np.float64)


def _build_log_times(
    form_factors: FormFactorsData,
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    return form_factors.scale * array_of_floats(
        support=form_factors.logD_time_support,
        num_pts=form_factors.num_times,
        log_base=form_factors.dimension,
    )


def _validate_real_series(
    form_factors: FormFactorsData,
    _: object,
    values: np.ndarray[tuple[int], np.dtype[np.floating]],
) -> None:
    if (
        values.shape != (form_factors.num_times,)
        or not np.issubdtype(values.dtype, np.floating)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("Form factor series does not match the declared time count.")


def _validate_complex_series(
    form_factors: FormFactorsData,
    _: object,
    values: np.ndarray[tuple[int], np.dtype[np.complexfloating]],
) -> None:
    if (
        values.shape != (form_factors.num_times,)
        or not np.issubdtype(values.dtype, np.complexfloating)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("Form factor series does not match the declared time count.")


def finalize_form_factors(form_factors: FormFactorsData) -> None:
    form_factors.compute_form_factors()


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class FormFactorsData(Data):
    data_type: ClassVar[str] = "form_factors"

    dimension: int = attrs.field(
        converter=int,
        validator=attrs.validators.gt(0),
    )
    logD_time_support: Support = attrs.field(
        default=(-1.5, 0.5),
        converter=to_support_pair,
        validator=is_valid_support,
    )
    scale: float = attrs.field(
        default=2 * np.pi,
        converter=float,
        validator=attrs.validators.gt(0.0),
    )
    num_times: int = attrs.field(
        default=NUM_TIMES,
        converter=int,
        validator=attrs.validators.gt(0),
    )
    time_chunk_size: int = attrs.field(
        default=TIME_CHUNK_SIZE,
        converter=int,
        validator=attrs.validators.gt(0),
    )

    times: np.ndarray[tuple[int], np.dtype[np.floating]] = attrs.field(
        default=attrs.Factory(_build_log_times, takes_self=True),
        converter=np.asarray,
        validator=_validate_real_series,
        repr=False,
    )
    first_moment: np.ndarray[tuple[int], np.dtype[np.complexfloating]] = attrs.field(
        default=attrs.Factory(_build_complex_zeros, takes_self=True),
        converter=np.asarray,
        validator=_validate_complex_series,
        repr=False,
    )
    second_moment: np.ndarray[tuple[int], np.dtype[np.floating]] = attrs.field(
        default=attrs.Factory(_build_float_zeros, takes_self=True),
        converter=np.asarray,
        validator=_validate_real_series,
        repr=False,
    )
    form_factor: np.ndarray[tuple[int], np.dtype[np.floating]] = attrs.field(
        default=attrs.Factory(_build_float_zeros, takes_self=True),
        converter=np.asarray,
        validator=_validate_real_series,
        repr=False,
    )
    connected_form_factor: np.ndarray[tuple[int], np.dtype[np.floating]] = attrs.field(
        default=attrs.Factory(_build_float_zeros, takes_self=True),
        converter=np.asarray,
        validator=_validate_real_series,
        repr=False,
    )
    realizs: int = attrs.field(
        default=0,
        converter=int,
        validator=attrs.validators.ge(0),
        repr=False,
    )

    @classmethod
    def create_raw(
        cls,
        *,
        ensemble: ManyBodyEnsemble,
        file_name_prefix: str,
    ) -> FormFactorsData:
        j_1_1 = cast(float, jn_zeros(1, 1)[0])

        raw_form_factors = FormFactorsData(
            file_name=f"{file_name_prefix}_data",
            dimension=ensemble.dimension,
            logD_time_support=LOG_D_TIME_SUPPORT,
            scale=j_1_1 / ensemble.spectral_radius,
        )
        metadata = {"unfolding": "raw"}
        raw_form_factors.attach_metadata(metadata)

        return raw_form_factors

    @classmethod
    def create_unfolded(
        cls,
        *,
        file_name_prefix: str,
        dimension: int,
        unfolding: str,
        degree: int | None = None,
    ) -> FormFactorsData:
        unfolded_form_factors = FormFactorsData(
            file_name=f"{file_name_prefix}_data",
            dimension=dimension,
            logD_time_support=LOG_D_UNFOLDED_TIME_SUPPORT,
            scale=2 * np.pi,
        )

        metadata: dict[str, int | str] = {"unfolding": unfolding}
        if degree is not None:
            metadata["degree"] = degree
        unfolded_form_factors.attach_metadata(metadata)

        return unfolded_form_factors

    def compute_moment_contributions(
        self,
        levels: np.ndarray[tuple[int], np.dtype[np.floating]],
        /,
    ) -> None:
        for start in range(0, len(self.times), self.time_chunk_size):
            stop = min(start + self.time_chunk_size, len(self.times))
            first_moment_contribution = cast(
                np.ndarray[tuple[int], np.dtype[np.complexfloating]],
                np.sum(
                    np.exp(-1j * np.outer(levels, self.times[start:stop])),
                    axis=0,
                )
                / len(levels),
            )

            self.first_moment[start:stop] += first_moment_contribution
            self.second_moment[start:stop] += np.abs(first_moment_contribution) ** 2

        object.__setattr__(self, "realizs", self.realizs + 1)

    def compute_form_factors(self) -> None:
        if self.realizs == 0:
            self.form_factor.fill(0.0)
            self.connected_form_factor.fill(0.0)
            return

        self.form_factor[:] = self.second_moment / self.realizs
        self.connected_form_factor[:] = (
            self.form_factor - np.abs(self.first_moment / self.realizs) ** 2
        )
