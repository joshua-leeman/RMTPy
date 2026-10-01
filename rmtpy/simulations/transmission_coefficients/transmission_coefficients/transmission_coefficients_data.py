from typing import Protocol, cast

import attrs
import numpy as np

from ...base_data import Data


class _TransmissionCoefficientsDataLike(Protocol):
    energies: np.ndarray[tuple[int], np.dtype[np.floating]]


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


def _build_complex_zeros(
    data: TransmissionCoefficientsData,
) -> np.ndarray[tuple[int], np.dtype[np.complexfloating]]:
    return np.zeros(len(data.energies), dtype=np.complex128)


def _build_float_zeros(
    data: TransmissionCoefficientsData,
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    return np.zeros(len(data.energies), dtype=np.float64)


def _normalize_complex_energy_vector(
    values: np.ndarray[tuple[int], np.dtype[np.complexfloating]],
    data: attrs.AttrsInstance,
) -> np.ndarray[tuple[int], np.dtype[np.complexfloating]]:
    data = cast(_TransmissionCoefficientsDataLike, data)
    array = np.array(values, dtype=np.complex128, copy=True, order="C")
    if array.shape != data.energies.shape or not np.all(np.isfinite(array)):
        raise ValueError(
            "Complex energy-resolved values must have shape "
            + f"{data.energies.shape} and contain only finite values."
        )

    return array


def _normalize_float_energy_vector(
    values: np.ndarray[tuple[int], np.dtype[np.floating]],
    data: attrs.AttrsInstance,
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    data = cast(_TransmissionCoefficientsDataLike, data)
    array = np.array(values, dtype=np.float64, copy=True, order="C")
    if array.shape != data.energies.shape or not np.all(np.isfinite(array)):
        raise ValueError(
            "Real energy-resolved values must have shape "
            + f"{data.energies.shape} and contain only finite values."
        )

    return array


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class TransmissionCoefficientsData(Data):
    energies: np.ndarray[tuple[int], np.dtype[np.floating]] = attrs.field(
        converter=_normalize_energies,
        repr=False,
    )
    channel_index: int = attrs.field(
        converter=int,
        validator=attrs.validators.ge(0),
    )

    scattering_diagonal_sum: np.ndarray[tuple[int], np.dtype[np.complexfloating]] = (
        attrs.field(
            default=attrs.Factory(_build_complex_zeros, takes_self=True),
            converter=attrs.Converter(_normalize_complex_energy_vector, takes_self=True),
            repr=False,
        )
    )
    average_scattering_diagonal: np.ndarray[tuple[int], np.dtype[np.complexfloating]] = (
        attrs.field(
            default=attrs.Factory(_build_complex_zeros, takes_self=True),
            converter=attrs.Converter(_normalize_complex_energy_vector, takes_self=True),
            repr=False,
        )
    )
    transmission_coefficients: np.ndarray[tuple[int], np.dtype[np.floating]] = (
        attrs.field(
            default=attrs.Factory(_build_float_zeros, takes_self=True),
            converter=attrs.Converter(_normalize_float_energy_vector, takes_self=True),
            repr=False,
        )
    )
    realizs: int = attrs.field(
        default=0,
        converter=int,
        validator=attrs.validators.ge(0),
    )

    @classmethod
    def create(
        cls,
        *,
        energies: np.ndarray[tuple[int], np.dtype[np.floating]],
        channel_index: int,
    ) -> TransmissionCoefficientsData:
        transmission_coefficients = TransmissionCoefficientsData(
            _file_name=f"transmission_coefficients_channel_{channel_index}",
            energies=energies,
            channel_index=channel_index,
        )

        metadata = {"channel_index": channel_index}
        transmission_coefficients.attach_metadata(metadata)

        return transmission_coefficients

    def add_scattering_diagonal(
        self,
        scattering_diagonal: np.ndarray[tuple[int], np.dtype[np.complexfloating]],
        /,
    ) -> None:
        scattering_diagonal = np.asarray(scattering_diagonal)
        if scattering_diagonal.shape != self.energies.shape:
            raise ValueError(
                "Energy-resolved scattering diagonal must have shape "
                + f"{self.energies.shape}, got {scattering_diagonal.shape}."
            )

        np.add(
            self.scattering_diagonal_sum,
            scattering_diagonal,
            out=self.scattering_diagonal_sum,
        )
        object.__setattr__(self, "realizs", self.realizs + 1)

    def compute_transmission_coefficients(self) -> None:
        if self.realizs == 0:
            self.average_scattering_diagonal.fill(0.0)
            self.transmission_coefficients.fill(0.0)
            return

        self.average_scattering_diagonal[:] = self.scattering_diagonal_sum / self.realizs
        self.transmission_coefficients[:] = (
            1.0 - np.abs(self.average_scattering_diagonal) ** 2
        )
        np.clip(
            self.transmission_coefficients,
            0.0,
            1.0,
            out=self.transmission_coefficients,
        )
