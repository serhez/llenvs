"""Small, strict value checks shared by the connector boundaries."""

import math
from collections.abc import Sequence
from numbers import Real


def integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite (overflow)") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite (overflow or non-finite input)")
    return number


def identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def token_ids(values: Sequence[int], name: str) -> None:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a sequence of integer token IDs")
    for value in values:
        integer(value, name)
