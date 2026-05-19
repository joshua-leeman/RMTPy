import numpy as np
from scipy.special import gamma

def eigval_degeneracy(dyson_index: int) -> int:
    return 2 if dyson_index == 4 else 1


def porter_thomas_distribution(
    dyson_index: float, num_channels: int, widths: np.ndarray
) -> np.ndarray:
    if dyson_index == 1:
        real_dof: int = num_channels
    else:
        real_dof: int = 2 * num_channels

    widths = np.asarray(widths)
    coeff: float = (real_dof / 2) ** (real_dof / 2) / gamma(real_dof / 2)
    return coeff * widths ** (real_dof / 2 - 1) * np.exp(-real_dof * widths / 2)


def universal_csff(dyson_index: float, dimension: int, times: np.ndarray) -> np.ndarray:
    tau: np.ndarray = np.asarray(times) / (2 * np.pi)

    if dyson_index == 1:
        csff: np.ndarray = np.empty_like(tau)

        mask: np.ndarray = tau <= 1
        csff[mask] = tau[mask] * (2 - np.log(2 * tau[mask] + 1)) / dimension
        csff[~mask] = (
            2 - tau[~mask] * np.log((2 * tau[~mask] + 1) / (2 * tau[~mask] - 1))
        ) / dimension

        return csff

    elif dyson_index == 2:
        return np.where(tau <= 1, tau / dimension, 1 / dimension)

    elif dyson_index == 4:
        csff: np.ndarray = np.full_like(tau, 2 / dimension)
        csff[2 * tau == 1] = np.nan

        mask: np.ndarray = (tau < 1) & (2 * tau != 1)
        csff[mask] = tau[mask] * (2 - np.log(np.abs(2 * tau[mask] - 1))) / dimension
        return csff

    else:
        return np.full_like(tau, 1 / dimension)


def universality_class(dyson_index: int) -> str | None:
    dyson_indices: dict[int, str] = {
        0: "Poisson",
        1: "GOE",
        2: "GUE",
        4: "GSE",
    }
    return dyson_indices.get(dyson_index, None)


def wigner_surmise(dyson_index: float, spacings: np.ndarray) -> np.ndarray:
    spacings = np.asarray(spacings)

    if dyson_index == 0:
        return np.exp(-spacings)

    degeneracy: int = eigval_degeneracy(dyson_index)
    adj_spacings: np.ndarray = spacings / degeneracy

    idx: float = dyson_index
    a: float = 2 * gamma((idx + 2) / 2) ** (idx + 1) / gamma((idx + 1) / 2) ** (idx + 2)
    b: float = ((gamma((idx + 2) / 2)) / gamma((idx + 1) / 2)) ** 2

    return a * adj_spacings**idx * np.exp(-b * adj_spacings**2) / degeneracy
