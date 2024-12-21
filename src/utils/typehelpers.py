from collections.abc import Callable
from functools import update_wrapper
from typing import cast


def inherit_signature_from[T, **P](
    original: Callable[P, T],
) -> Callable[[Callable], Callable[P, T]]:
    """Set the signature of one function to the signature of another."""

    def wrapper(f: Callable) -> Callable[P, T]:
        return cast(Callable[P, T], update_wrapper(f, original))

    return wrapper
