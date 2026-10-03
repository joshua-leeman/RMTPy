from typing import Protocol, cast

import attrs
import numpy as np

from ...base_data import Data


class _WeisskopfEstimateDataLike(Protocol):
    energies: np.ndarray[tuple[int], np.dtype[np.floating]]
    num_channels: int


def _normalize_energies(
    energies: np.ndarray[tuple[int], np.dtype[np.floating]],
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    energies_array = np.array(energies, dtype=np.float64, copy=True, order="C")
    if energies_array.ndim != 1 or energies_array.size < 2:
        raise ValueError(
            "`energies` must be a one-dimensional array with at least two entries."
        )
    if not np.all(np.isfinite(energies_array)):
        raise ValueError("`energies` must contain finite values.")
    if np.any(np.diff(energies_array) <= 0.0):
        raise ValueError("`energies` must be strictly increasing.")

    energies_array.flags.writeable = False
    return energies_array


def _normalize_mean_level_spacings(
    mean_level_spacings: np.ndarray[tuple[int], np.dtype[np.floating]],
    data: attrs.AttrsInstance,
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    data = cast(_WeisskopfEstimateDataLike, data)
    spacings = np.array(
        mean_level_spacings,
        dtype=np.float64,
        copy=True,
        order="C",
    )
    if spacings.shape != data.energies.shape:
        raise ValueError(
            "`mean_level_spacings` must have shape "
            + f"{data.energies.shape}, got {spacings.shape}."
        )
    if np.any(np.isinf(spacings)) or np.any(spacings[np.isfinite(spacings)] <= 0.0):
        raise ValueError(
            "Finite `mean_level_spacings` values must be positive, and undefined "
            + "values must be represented by NaN."
        )

    spacings.flags.writeable = False
    return spacings


def _build_complex_matrix(
    data: WeisskopfEstimateData,
) -> np.ndarray[tuple[int, int], np.dtype[np.complexfloating]]:
    return np.zeros(
        (len(data.energies), data.num_channels),
        dtype=np.complex128,
    )


def _build_float_matrix(
    data: WeisskopfEstimateData,
) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
    return np.zeros(
        (len(data.energies), data.num_channels),
        dtype=np.float64,
    )


def _build_float_vector(
    data: WeisskopfEstimateData,
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    return np.zeros(len(data.energies), dtype=np.float64)


def _normalize_complex_channel_matrix(
    values: np.ndarray[tuple[int, int], np.dtype[np.complexfloating]],
    data: attrs.AttrsInstance,
) -> np.ndarray[tuple[int, int], np.dtype[np.complexfloating]]:
    data = cast(_WeisskopfEstimateDataLike, data)
    array = np.array(values, dtype=np.complex128, copy=True, order="C")
    expected_shape = (len(data.energies), data.num_channels)
    if array.shape != expected_shape or not np.all(np.isfinite(array)):
        raise ValueError(
            "Complex channel-resolved values must have shape "
            + f"{expected_shape} and contain only finite values."
        )

    return array


def _normalize_float_channel_matrix(
    values: np.ndarray[tuple[int, int], np.dtype[np.floating]],
    data: attrs.AttrsInstance,
) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
    data = cast(_WeisskopfEstimateDataLike, data)
    array = np.array(values, dtype=np.float64, copy=True, order="C")
    expected_shape = (len(data.energies), data.num_channels)
    if array.shape != expected_shape or not np.all(np.isfinite(array)):
        raise ValueError(
            "Real channel-resolved values must have shape "
            + f"{expected_shape} and contain only finite values."
        )

    return array


def _normalize_weisskopf_estimate(
    values: np.ndarray[tuple[int], np.dtype[np.floating]],
    data: attrs.AttrsInstance,
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    data = cast(_WeisskopfEstimateDataLike, data)
    array = np.array(values, dtype=np.float64, copy=True, order="C")
    if array.shape != data.energies.shape or np.any(np.isinf(array)):
        raise ValueError(
            "`weisskopf_estimate` must match the energy grid and may contain only "
            + "finite values or NaN."
        )

    return array


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class WeisskopfEstimateData(Data):
    energies: np.ndarray[tuple[int], np.dtype[np.floating]] = attrs.field(
        converter=_normalize_energies,
        repr=False,
    )
    mean_level_spacings: np.ndarray[tuple[int], np.dtype[np.floating]] = attrs.field(
        converter=attrs.Converter(_normalize_mean_level_spacings, takes_self=True),
        repr=False,
    )
    num_channels: int = attrs.field(
        converter=int,
        validator=attrs.validators.gt(0),
    )

    scattering_diagonal_sum: np.ndarray[tuple[int, int], np.dtype[np.complexfloating]] = (
        attrs.field(
            default=attrs.Factory(_build_complex_matrix, takes_self=True),
            converter=attrs.Converter(_normalize_complex_channel_matrix, takes_self=True),
            repr=False,
        )
    )
    average_scattering_diagonal: np.ndarray[
        tuple[int, int], np.dtype[np.complexfloating]
    ] = attrs.field(
        default=attrs.Factory(_build_complex_matrix, takes_self=True),
        converter=attrs.Converter(_normalize_complex_channel_matrix, takes_self=True),
        repr=False,
    )
    transmission_coefficients: np.ndarray[tuple[int, int], np.dtype[np.floating]] = (
        attrs.field(
            default=attrs.Factory(_build_float_matrix, takes_self=True),
            converter=attrs.Converter(_normalize_float_channel_matrix, takes_self=True),
            repr=False,
        )
    )
    weisskopf_estimate: np.ndarray[tuple[int], np.dtype[np.floating]] = attrs.field(
        default=attrs.Factory(_build_float_vector, takes_self=True),
        converter=attrs.Converter(_normalize_weisskopf_estimate, takes_self=True),
        repr=False,
    )

    @classmethod
    def create(
        cls,
        *,
        energies: np.ndarray[tuple[int], np.dtype[np.floating]],
        mean_level_spacings: np.ndarray[tuple[int], np.dtype[np.floating]],
        num_channels: int,
    ) -> WeisskopfEstimateData:
        weisskopf_estimate = WeisskopfEstimateData(
            _file_name="weisskopf_estimate",
            energies=energies,
            mean_level_spacings=mean_level_spacings,
            num_channels=num_channels,
        )

        metadata = {"num_channels": num_channels}
        weisskopf_estimate.attach_metadata(metadata)

        return weisskopf_estimate

    def add_scattering_diagonal(
        self,
        scattering_diagonal: np.ndarray[tuple[int, int], np.dtype[np.complexfloating]],
        /,
    ) -> None:
        scattering_diagonal = np.asarray(scattering_diagonal)
        expected_shape = (len(self.energies), self.num_channels)
        if scattering_diagonal.shape != expected_shape:
            raise ValueError(
                "Complete scattering diagonal must have shape "
                + f"{expected_shape}, got {scattering_diagonal.shape}."
            )

        _ = np.add(
            self.scattering_diagonal_sum,
            scattering_diagonal,
            out=self.scattering_diagonal_sum,
        )

        object.__setattr__(self, "realizs", self.realizs + 1)

    def compute_weisskopf_estimate(self) -> None:
        if self.realizs == 0:
            self.average_scattering_diagonal.fill(0.0)
            self.transmission_coefficients.fill(0.0)
        else:
            self.average_scattering_diagonal[:] = (
                self.scattering_diagonal_sum / self.realizs
            )
            self.transmission_coefficients[:] = (
                1.0 - np.abs(self.average_scattering_diagonal) ** 2
            )
            _ = np.clip(
                self.transmission_coefficients,
                0.0,
                1.0,
                out=self.transmission_coefficients,
            )

        self.weisskopf_estimate[:] = self.mean_level_spacings / 2.0

        channel_sum = cast(
            np.ndarray[tuple[int], np.dtype[np.floating]],
            np.sum(self.transmission_coefficients, axis=1),
        )
        _ = np.multiply(
            self.weisskopf_estimate,
            channel_sum,
            out=self.weisskopf_estimate,
        )

    def add_contribution(self, contribution: Data, /) -> None:
        self._validate_contribution(contribution)
        if not isinstance(contribution, WeisskopfEstimateData):
            raise TypeError("Weisskopf-estimate contribution is malformed.")
        if (
            contribution.num_channels != self.num_channels
            or not np.array_equal(contribution.energies, self.energies)
            or not np.array_equal(
                contribution.mean_level_spacings,
                self.mean_level_spacings,
                equal_nan=True,
            )
        ):
            raise ValueError(
                f"Weisskopf-estimate contribution `{self._file_name}` has "
                + "incompatible static data."
            )

        self.scattering_diagonal_sum[:] += contribution.scattering_diagonal_sum
        self._add_realizations(contribution)

    def compute_statistics(self) -> None:
        self.compute_weisskopf_estimate()
