# Standard Library
import os
from contextlib import contextmanager
from io import RawIOBase
from typing import IO, AnyStr, Optional

# Import from alt
from alt.files.interfaces import Readable, Seekable, Writable
from alt.files.utils import (
    get_content_size,
    get_mode,
    get_name,
    is_readable,
    is_writable,
)


class BaseShadowHandler(RawIOBase):
    """ShadowHandler using RawIOBase's interface. (avoid type checking error)"""


class ShadowHandler(Readable, Seekable, Writable, BaseShadowHandler):
    """Create a File-Like Object, maintaining file pointer, to avoid misunderstanding the position when read / write / seek
    It can be roughly regarded as the copy function of the file handle, but you need to be careful with the write handle, because no matter which copy will modify the data itself
    """

    def __init__(self, file_object: IO, intrusive: bool = True):
        self._file_object = file_object
        self._offset = file_object.tell()
        self._intrusive = intrusive

    @property
    def name(self) -> str:
        return get_name(self._file_object)

    @property
    def mode(self) -> str:
        return get_mode(self._file_object)

    @contextmanager
    def _ensure_offset(self):
        offset = self._file_object.tell()
        if offset != self._offset:
            self._file_object.seek(self._offset)
        yield
        self._offset = self._file_object.tell()
        if not self._intrusive:
            self._file_object.seek(offset)

    @property
    def _content_size(self) -> int:
        return get_content_size(self._file_object, intrusive=self._intrusive)

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            self._offset = offset
        elif whence == os.SEEK_CUR:
            self._offset = self._offset + offset
        elif whence == os.SEEK_END:
            self._offset = self._content_size + offset
        return self._offset

    def tell(self) -> int:
        return self._offset

    def readable(self) -> bool:
        return is_readable(self._file_object)

    def read(self, size: Optional[int] = None) -> AnyStr:  # pytype: disable=signature-mismatch
        with self._ensure_offset():
            return self._file_object.read(size)

    def readline(self, size: Optional[int] = None) -> AnyStr:  # pytype: disable=signature-mismatch
        with self._ensure_offset():
            return self._file_object.readline(size)

    def writable(self) -> bool:
        return is_writable(self._file_object)

    def write(self, data: AnyStr):  # pytype: disable=signature-mismatch
        with self._ensure_offset():
            return self._file_object.write(data)

    def _close(self):
        pass
