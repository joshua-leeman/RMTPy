def validate_even_number(number: int) -> None:
    if number % 2 != 0:
        raise ValueError("`number` must be an even integer.")


def validate_support(support: tuple[float, float]) -> None:
    support = tuple(support)
    if len(support) != 2:
        raise ValueError("`support` must have length 2.")
    if support[0] >= support[1]:
        raise ValueError("`support` must be strictly increasing.")
