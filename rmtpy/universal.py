from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.special import gamma


def eigval_degeneracy(*, dyson_index: int) -> int:
    return 2 if dyson_index == 4 else 1


def universality_class(*, dyson_index: int) -> str | None:
    return {0: "Poisson", 1: "GOE", 2: "GUE", 4: "GSE"}.get(dyson_index)


def wigner_surmise(
    spacings: NDArray[np.float64],
    /,
    *,
    dyson_index: int,
) -> NDArray[np.float64]:
    if dyson_index == 0:
        return np.exp(-spacings)

    degeneracy = eigval_degeneracy(dyson_index=dyson_index)
    adjusted_spacings = spacings / degeneracy

    idx = dyson_index
    a = 2 * gamma((idx + 2) / 2) ** (idx + 1) / gamma((idx + 1) / 2) ** (idx + 2)
    b = ((gamma((idx + 2) / 2)) / gamma((idx + 1) / 2)) ** 2

    return a * adjusted_spacings**idx * np.exp(-b * adjusted_spacings**2) / degeneracy


def porter_thomas_distribution(
    widths: NDArray[np.float64],
    /,
    *,
    dyson_index: int,
    num_channels: int,
) -> NDArray[np.float64]:
    real_dof = num_channels if dyson_index == 1 else 2 * num_channels
    coeff = cast(int, (real_dof / 2) ** (real_dof / 2) / gamma(real_dof / 2))
    return coeff * widths ** (real_dof / 2 - 1) * np.exp(-real_dof * widths / 2)


def connected_sff(
    times: NDArray[np.float64],
    /,
    *,
    dyson_index: int,
    dimension: int,
) -> NDArray[np.float64]:
    tau = times / (2 * np.pi)

    if dyson_index == 1:
        csff = np.empty_like(tau)

        mask = tau <= 1
        csff[mask] = tau[mask] * (2 - np.log(2 * tau[mask] + 1)) / dimension
        csff[~mask] = (
            2 - tau[~mask] * np.log((2 * tau[~mask] + 1) / (2 * tau[~mask] - 1))
        ) / dimension

        return csff

    elif dyson_index == 2:
        return np.where(tau <= 1, tau / dimension, 1 / dimension)

    elif dyson_index == 4:
        csff = np.full_like(tau, 2 / dimension)
        csff[2 * tau == 1] = np.nan

        mask = cast(NDArray[np.bool_], (tau < 1) & (2 * tau != 1))
        csff[mask] = tau[mask] * (2 - np.log(np.abs(2 * tau[mask] - 1))) / dimension

        return csff

    else:
        return np.full_like(tau, 1 / dimension)


def time_delay_pdf(
    times: NDArray[np.float64],
    /,
    *,
    num_channels: int,
    heisenberg_time: float,
) -> NDArray[np.float64]:
    taus = times / heisenberg_time
    tau_plus = cast(float, (3 + np.sqrt(8)) / num_channels)
    tau_minus = cast(float, (3 - np.sqrt(8)) / num_channels)

    pdf = np.zeros_like(times, dtype=np.result_type(times.dtype, np.float64))
    in_support = (tau_minus < taus) & (taus < tau_plus)
    pdf[in_support] = np.sqrt(
        (tau_plus - taus[in_support]) * (taus[in_support] - tau_minus)
    ) / (2 * np.pi * taus[in_support] ** 2 * heisenberg_time)

    return pdf
