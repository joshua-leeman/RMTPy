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


def create_complex_zeros(data: TransmissionCoefficientsData) -> np.ndarray:
    return np.zeros(len(data.energies), dtype=np.complex128)


def create_float_zeros(data: TransmissionCoefficientsData) -> np.ndarray:
    return np.zeros(len(data.energies), dtype=np.float64)


def create_realizs_count(_: TransmissionCoefficientsData) -> np.ndarray:
    return np.zeros(1, dtype=np.int64)


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class TransmissionCoefficientsData(Data):
    """Running energy-resolved average for one diagonal scattering channel."""

    energies: np.ndarray = attrs.field(converter=normalize_energies, repr=False)
    channel_index: int = attrs.field(converter=int)

    scattering_diagonal_sum: np.ndarray = attrs.field(
        default=attrs.Factory(create_complex_zeros, takes_self=True),
        init=False,
        repr=False,
    )
    average_scattering_diagonal: np.ndarray = attrs.field(
        default=attrs.Factory(create_complex_zeros, takes_self=True),
        init=False,
        repr=False,
    )
    transmission_coefficients: np.ndarray = attrs.field(
        default=attrs.Factory(create_float_zeros, takes_self=True),
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
        expected_shape = self.energies.shape
        if scattering_diagonal.shape != expected_shape:
            raise ValueError(
                "Energy-resolved scattering diagonal must have shape "
                f"{expected_shape}, got {scattering_diagonal.shape}."
            )

        np.add(
            self.scattering_diagonal_sum,
            scattering_diagonal,
            out=self.scattering_diagonal_sum,
        )
        self._realizs_count[0] += 1

    def compute_transmission_coefficients(self) -> None:
        if self.realizs == 0:
            self.average_scattering_diagonal.fill(0.0)
            self.transmission_coefficients.fill(0.0)
            return

        self.average_scattering_diagonal[:] = (
            self.scattering_diagonal_sum / self.realizs
        )
        self.transmission_coefficients[:] = (
            1.0 - np.abs(self.average_scattering_diagonal) ** 2
        )
        np.clip(self.transmission_coefficients, 0.0, 1.0, out=self.transmission_coefficients)
