import math
from collections.abc import Sequence
from typing import Any


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


def is_even_number(_: Any, __: Any, number: int) -> None:
    validate_even_number(number)


def is_valid_support(_: Any, __: Any, support: Sequence[float]) -> None:
    validate_support(support)
