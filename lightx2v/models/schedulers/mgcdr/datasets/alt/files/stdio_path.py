# Standard Library
import io
from typing import IO, AnyStr, Optional, Union

# Import from alt
from alt.files.interfaces import BaseURIPath, PathLike
from alt.files.lib.compat import fspath
from alt.files.lib.stdio_handler import STDReader, STDWriter
from alt.files.lib.url import get_url_scheme
from alt.files.smart_path import SmartPath
from alt.files.utils import get_binary_mode

__all__ = [
    "StdioPath",
    "is_stdio",
]


def is_stdio(path: PathLike) -> bool:
    """stdio scheme definition: stdio://-

    .. note ::

        Only tests protocol

    :param path: Path to be tested
    :returns: True of a path is stdio url, else False
    """

    path = fspath(path)
    if not isinstance(path, str) or not path.startswith("stdio://"):
        return False

    scheme = get_url_scheme(path)
    return scheme == "stdio"


@SmartPath.register
class StdioPath(BaseURIPath):

    protocol = "stdio"

    def _open(self, mode: str = "rb") -> Union[STDReader, STDWriter]:
        """Used to read or write stdio

        .. note ::

            Essentially invoke sys.stdin.buffer | sys.stdout.buffer to read or write

        :param path: stdio path, stdio://- or stdio://0 stdio://1 stdio://2
        :param mode: Only supports 'rb' and 'wb' now
        :return: STDReader, STDWriter
        """

        if mode not in ("rb", "wb", "rt", "wt", "r", "w"):
            raise ValueError("unacceptable mode: %r" % mode)

        mode = get_binary_mode(mode)

        if self.path_with_protocol not in ("stdio://-", "stdio://0", "stdio://1", "stdio://2"):
            raise ValueError("unacceptable path: %r" % self.path_with_protocol)

        if self.path_with_protocol in ("stdio://1", "stdio://2") and "r" in mode:
            raise ValueError("cannot open for reading: %r" % self.path_with_protocol)

        if self.path_with_protocol == "stdio://0" and "w" in mode:
            raise ValueError("cannot open for writing: %r" % self.path_with_protocol)

        if "r" in mode:
            return STDReader(mode)
        return STDWriter(self.path_with_protocol, mode)

    def open(
        self, mode: str = "rb", encoding: Optional[str] = None, errors: Optional[str] = None, **kwargs
    ) -> IO[AnyStr]:  # pytype: disable=signature-mismatch
        """Used to read or write stdio

        .. note ::

            Essentially invoke sys.stdin.buffer | sys.stdout.buffer to read or write

        :param mode: Only supports 'rb' and 'wb' now
        :return: STDReader, STDWriter
        """
        fileobj = self._open(mode)

        if "b" not in mode:
            fileobj = io.TextIOWrapper(fileobj, encoding=encoding, errors=errors)  # type: ignore
            fileobj.mode = mode

        return fileobj  # type: ignore
