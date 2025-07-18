# Standard Library
from typing import IO, AnyStr, Optional

# Import from alt
from alt.files.interfaces import PathLike
from alt.files.stdio_path import StdioPath, is_stdio

__all__ = [
    "is_stdio",
    "stdio_open",
]


def stdio_open(
    path: PathLike, mode: str = "rb", encoding: Optional[str] = None, errors: Optional[str] = None, **kwargs
) -> IO[AnyStr]:  # pytype: disable=signature-mismatch
    """Used to read or write stdio

    .. note ::

        Essentially invoke sys.stdin.buffer | sys.stdout.buffer to read or write

    :param path: Given path
    :param mode: Only supports 'rb' and 'wb' now
    :return: STDReader, STDWriter
    """
    return StdioPath(path).open(mode, encoding, errors)
