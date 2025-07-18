# Standard Library
import os
from abc import ABC, abstractmethod
from io import UnsupportedOperation
from typing import Iterable, Iterator, List, Optional

# Import from alt
from alt.files.pathlike import Access, BasePath, BaseURIPath, FileEntry, PathLike, StatResult, URIPath  # noqa


def fullname(o):
    klass = o.__class__
    module = klass.__module__
    if module == "builtins":
        return klass.__qualname__  # avoid outputs like 'builtins.str'
    return module + "." + klass.__qualname__


# 1. Default value of closed is False
# 2. closed is set to True when close() are called
# 3. close() will only be called once
class Closable(ABC):

    @property
    def closed(self) -> bool:
        """Return True if the file-like object is closed."""
        return getattr(self, "__closed__", False)

    @abstractmethod
    def _close(self) -> None:
        pass  # pragma: no cover

    def close(self) -> None:
        """Flush and close the file-like object.

        This method has no effect if the file is already closed.
        """
        if not getattr(self, "__closed__", False):
            self._close()
            setattr(self, "__closed__", True)

    def __enter__(self) -> "Closable":
        return self

    def __exit__(self, type, value, traceback) -> None:
        self.close()


class FileLike(Closable, ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass  # pragma: no cover

    @property
    @abstractmethod
    def mode(self) -> str:
        pass  # pragma: no cover

    def fileno(self) -> int:
        raise UnsupportedOperation("not a local file")

    def __repr__(self) -> str:
        return "<%s name=%r mode=%r>" % (fullname(self), self.name, self.mode)  # pragma: no cover

    def seekable(self) -> bool:
        """Return True if the file-like object can be seeked."""
        return False

    def seek(self, cookie: int, whence: int = os.SEEK_SET) -> int:
        """Change stream position.

        Seek to byte offset `cookie` relative to position indicated by `whence`:
            0  Start of stream (the default).  `cookie` should be >= 0;
            1  Current position - `cookie` may be negative;
            2  End of stream - `cookie` usually negative.

        Return the new absolute position.
        """
        raise UnsupportedOperation("not seekable")  # pragma: no cover

    @abstractmethod
    def tell(self) -> int:
        """Return current stream position."""

    def readable(self) -> bool:
        """Return True if the file-like object can be read."""
        return False  # pragma: no cover

    def writable(self) -> bool:
        """Return True if the file-like object can be written."""
        return False

    def flush(self) -> None:
        """Flush write buffers, if applicable.

        This is not implemented for read-only and non-blocking streams.
        """


class Seekable(FileLike, ABC):

    def seekable(self) -> bool:
        """Return True if the file-like object can be seeked."""
        return True

    @abstractmethod
    def seek(self, cookie: int, whence: int = os.SEEK_SET) -> int:
        """Change stream position.

        Seek to byte offset `cookie` relative to position indicated by `whence`:
            0  Start of stream (the default).  `cookie` should be >= 0;
            1  Current position - `cookie` may be negative;
            2  End of stream - `cookie` usually negative.

        Return the new absolute position.
        """


class Readable(FileLike, ABC):

    def readable(self) -> bool:
        """Return True if the file-like object can be read."""
        return True

    @abstractmethod
    def read(self, size: Optional[int] = None) -> bytes:
        """Read at most `size` bytes, returned as a bytes object.

        If the `size` argument is negative, read until EOF is reached.
        Return an empty bytes object at EOF.
        """

    @abstractmethod
    def readline(self, size: Optional[int] = None) -> bytes:
        """Next line from the file, as a bytes object.

        Retain newline. A non-negative `size` argument limits the maximum number of bytes to return (an incomplete line may be returned then).
        Return an empty bytes object at EOF.
        """

    def readlines(self) -> List[bytes]:
        """Return a list of lines from the stream."""
        return self.read().splitlines(True)

    def readinto(self, buffer: bytearray) -> int:
        """Read bytes into buffer.

        Returns number of bytes read (0 for EOF), or None if the object
        is set not to block and has no data to read.
        """
        data = self.read(len(buffer))
        size = len(data)
        buffer[:size] = data
        return size

    def __next__(self) -> bytes:
        line = self.readline()
        if not line:
            raise StopIteration
        return line

    def __iter__(self) -> Iterator[bytes]:
        return self


class Writable(FileLike, ABC):

    def writable(self) -> bool:
        """Return True if the file-like object can be written."""
        return True

    @abstractmethod
    def write(self, data: bytes) -> int:
        """Write bytes to file.

        Return the number of bytes written.
        """

    def writelines(self, lines: Iterable[bytes]) -> None:
        """Write `lines` to the file.

        Note that newlines are not added. `lines` can be any iterable object producing bytes-like objects. This is equivalent to calling write() for each element.
        """
        for line in lines:
            self.write(line)


class FileCacher(Closable):

    @property
    @abstractmethod
    def cache_path(self) -> str:
        pass  # pragma: no cover

    def __enter__(self) -> str:  # pytype: disable=signature-mismatch
        return self.cache_path

    def __del__(self):
        self.close()


class NullCacher(FileCacher):
    cache_path = None

    def __init__(self, path):
        self.cache_path = path

    def _close(self):
        pass


class ContextIterator(Closable):

    def __init__(self, iterable: Iterable) -> None:
        self._iter = iter(iterable)

    def _close(self) -> None:
        pass

    def __next__(self):
        return next(self._iter)

    def __iter__(self):
        return self
