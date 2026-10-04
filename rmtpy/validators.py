import math
from collections.abc import Iterable, Sequence
from typing import cast

import numpy as np


def to_increasing_energy_grid(
    energies: np.ndarray[tuple[int], np.dtype[np.floating]],
    /,
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


def to_support_pair(support: Iterable[float], /) -> tuple[float, float]:
    values = tuple(support)
    if len(values) != 2:
        raise ValueError("`support` must have length 2.")

    first, second = values
    return (float(first), float(second))


def validate_even_number(number: int) -> None:
    if number % 2 != 0:
        raise ValueError("`number` must be an even integer.")


def validate_support(support: Sequence[float]) -> None:
    if len(support) != 2:
        raise ValueError("`support` must have length 2.")
    if not all(math.isfinite(endpoint) for endpoint in support):
        raise ValueError("`support` endpoints must be finite.")
    if support[0] >= support[1]:
        raise ValueError("`support` must be strictly increasing.")


def is_even_number(_inst: object, _attr: object, number: int) -> None:
    validate_even_number(number)


def is_support(_inst: object, _attr: object, support: Sequence[float]) -> None:
    validate_support(support)


def is_source_dict(_inst: object, _attr: object, configuration: object) -> None:
    if not isinstance(configuration, dict):
        raise TypeError("`configuration` must be a dict.")

    mapping = cast(dict[object, object], configuration)
    for key, item in mapping.items():
        if not isinstance(key, str):
            raise TypeError("`configuration` keys must be strings.")
        if isinstance(item, str):
            continue
        if not isinstance(item, dict):
            raise TypeError("`configuration` values must be strings or dicts.")

        nested = cast(dict[object, object], item)
        for nested_key in nested:
            if not isinstance(nested_key, str):
                raise TypeError("`configuration` nested keys must be strings.")
