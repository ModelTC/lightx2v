# Standard Library
import sys
from io import UnsupportedOperation
from typing import AnyStr, Optional

# Import from alt
from alt.files.interfaces import Readable, Writable


class STDHandler:

    def tell(self):
        raise UnsupportedOperation("not tellable")

    def _close(self):
        pass


class STDReader(STDHandler, Readable):
    """alt.file encapsulation of stdin. Avoid direct operation on sys.stdin

    .. note ::

        1. For convenience, use buffer by default

        2. There is currently no demand and no design for seek, so seek is not allowed now
    """

    def __init__(self, mode: str):
        handler = sys.stdin
        if "b" in mode:
            handler = handler.buffer

        self._handler = handler
        self._mode = mode

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def name(self) -> str:
        return "stdin"

    def read(self, size: Optional[int] = None) -> AnyStr:  # pytype: disable=signature-mismatch
        return self._handler.read(size)

    def readline(self, size: Optional[int] = None) -> AnyStr:  # pytype: disable=signature-mismatch
        return self._handler.readline()


class STDWriter(STDHandler, Writable):
    """alt.file encapsulation of stdin. Avoid direct operation on sys.stdin

    .. note ::

        1. For convenience, use buffer by default

        2. There is currently no demand and no design for seek, so seek is not allowed now
    """

    def __init__(self, path: str, mode: str):
        if path == "stdio://2":
            name = "stderr"
            handler = sys.stderr
        else:
            name = "stdout"
            handler = sys.stdout
        if "b" in mode:
            handler = handler.buffer

        self._handler = handler
        self._name = name
        self._mode = mode

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def name(self) -> str:
        return self._name

    def write(self, data: AnyStr) -> int:  # pytype: disable=signature-mismatch
        return self._handler.write(data)
