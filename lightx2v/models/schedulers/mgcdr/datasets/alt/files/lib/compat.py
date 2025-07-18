# Standard Library
import os
from os import PathLike

__all__ = [
    "PathLike",
    "fspath",
]


def fspath(path) -> str:
    result = os.fspath(path)
    if isinstance(result, bytes):
        return result.decode()
    return result
