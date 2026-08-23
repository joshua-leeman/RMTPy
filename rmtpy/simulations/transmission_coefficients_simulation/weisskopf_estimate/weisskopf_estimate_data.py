from __future__ import annotations

import attrs
import numpy as np

from ...data import Data


def normalize_energies(energies: np.ndarray) -> np.ndarray:
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


def normalize_mean_level_spacings(
    mean_level_spacings: np.ndarray,
    data: WeisskopfEstimateData,
) -> np.ndarray:
    spacings = np.array(
        mean_level_spacings,
        dtype=np.float64,
        copy=True,
        order="C",
    )
    if spacings.shape != data.energies.shape:
        raise ValueError(
            "`mean_level_spacings` must have shape "
            f"{data.energies.shape}, got {spacings.shape}."
        )
    if np.any(np.isinf(spacings)) or np.any(spacings[np.isfinite(spacings)] <= 0.0):
        raise ValueError(
            "Finite `mean_level_spacings` values must be positive, and undefined "
            "values must be represented by NaN."
        )
    spacings.flags.writeable = False
    return spacings


def create_complex_matrix(data: WeisskopfEstimateData) -> np.ndarray:
    return np.zeros(
        (len(data.energies), data.num_channels),
        dtype=np.complex128,
    )


def create_float_matrix(data: WeisskopfEstimateData) -> np.ndarray:
    return np.zeros(
        (len(data.energies), data.num_channels),
        dtype=np.float64,
    )


def create_float_vector(data: WeisskopfEstimateData) -> np.ndarray:
    return np.zeros(len(data.energies), dtype=np.float64)


def create_realizs_count(_: WeisskopfEstimateData) -> np.ndarray:
    return np.zeros(1, dtype=np.int64)


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class WeisskopfEstimateData(Data):
    """Running Weisskopf-width estimate using every open channel."""

    energies: np.ndarray = attrs.field(converter=normalize_energies, repr=False)
    mean_level_spacings: np.ndarray = attrs.field(
        converter=attrs.Converter(normalize_mean_level_spacings, takes_self=True),
        repr=False,
    )
    num_channels: int = attrs.field(
        converter=int,
        validator=attrs.validators.gt(0),
    )

    scattering_diagonal_sum: np.ndarray = attrs.field(
        default=attrs.Factory(create_complex_matrix, takes_self=True),
        init=False,
        repr=False,
    )
    average_scattering_diagonal: np.ndarray = attrs.field(
        default=attrs.Factory(create_complex_matrix, takes_self=True),
        init=False,
        repr=False,
    )
    transmission_coefficients: np.ndarray = attrs.field(
        default=attrs.Factory(create_float_matrix, takes_self=True),
        init=False,
        repr=False,
    )
    weisskopf_estimate: np.ndarray = attrs.field(
        default=attrs.Factory(create_float_vector, takes_self=True),
        init=False,
        repr=False,
    )
    _realizs_count: np.ndarray = attrs.field(
        default=attrs.Factory(create_realizs_count, takes_self=True),
        init=False,
        repr=False,
    )

    @property
    def realizs(self) -> int:
        return int(self._realizs_count[0])

    def add_scattering_diagonal(self, scattering_diagonal: np.ndarray) -> None:
        scattering_diagonal = np.asarray(scattering_diagonal)
        expected_shape = (len(self.energies), self.num_channels)
        if scattering_diagonal.shape != expected_shape:
            raise ValueError(
                "Complete scattering diagonal must have shape "
                f"{expected_shape}, got {scattering_diagonal.shape}."
            )

        np.add(
            self.scattering_diagonal_sum,
            scattering_diagonal,
            out=self.scattering_diagonal_sum,
        )
        self._realizs_count[0] += 1

    def compute_weisskopf_estimate(self) -> None:
        if self.realizs == 0:
            self.average_scattering_diagonal.fill(0.0)
            self.transmission_coefficients.fill(0.0)
            self.weisskopf_estimate.fill(0.0)
            return

        self.average_scattering_diagonal[:] = (
            self.scattering_diagonal_sum / self.realizs
        )
        self.transmission_coefficients[:] = (
            1.0 - np.abs(self.average_scattering_diagonal) ** 2
        )
        np.clip(self.transmission_coefficients, 0.0, 1.0, out=self.transmission_coefficients)

        self.weisskopf_estimate[:] = self.mean_level_spacings / 2.0
        np.multiply(
            self.weisskopf_estimate,
            np.sum(self.transmission_coefficients, axis=1),
            out=self.weisskopf_estimate,
        )
