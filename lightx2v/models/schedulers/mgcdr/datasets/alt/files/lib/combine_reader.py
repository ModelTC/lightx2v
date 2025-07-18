# Standard Library
import os
from io import BytesIO, StringIO
from typing import IO, AnyStr, List, Optional, Union

# Import from alt
from alt.files.interfaces import Readable, Seekable
from alt.files.utils import get_content_size, get_mode, get_name, is_readable

NEWLINE = ord("\n")


class CombineReader(Readable, Seekable):

    def __init__(self, file_objects: List[IO], name: str):
        self._file_objects = file_objects
        self._blocks_sizes = []
        self._content_size = 0
        self._offset = 0
        self._name = name
        self._mode = None
        for file_object in self._file_objects:
            if not is_readable(file_object):
                raise IOError("not readable: %r" % get_name(file_object))
            mode = get_mode(file_object)
            if self._mode is None:
                self._mode = mode
            if self._mode != mode:
                raise IOError("inconsistent mode: %r, expected: %r, got: %r" % (get_name(file_object), self._mode, mode))
            self._blocks_sizes.append(self._content_size)
            self._content_size += get_content_size(file_object)
        self._blocks_sizes.append(self._content_size)

    @property
    def _block_index_and_offset(self):
        for index, size in enumerate(self._blocks_sizes):
            if self._offset < size:
                return index - 1, self._offset - self._blocks_sizes[index - 1]
        raise IOError("offset out of range: %d" % self._offset)

    @property
    def name(self) -> str:
        return self._name

    @property
    def mode(self) -> str:
        return self._mode

    def tell(self) -> int:
        return self._offset

    def _empty_bytes(self) -> AnyStr:  # pytype: disable=signature-mismatch
        if "b" in self._mode:
            return b""
        return ""

    def _empty_buffer(self) -> Union[BytesIO, StringIO]:
        if "b" in self._mode:
            return BytesIO()
        return StringIO()

    def read(self, size: Optional[int] = None) -> AnyStr:  # pytype: disable=signature-mismatch
        if self._offset >= self._content_size:
            return self._empty_bytes()
        if size is None or size < 0:
            size = self._content_size - self._offset
        buffer = self._empty_buffer()
        while size > 0 and self._offset < self._content_size:
            block_index, block_offset = self._block_index_and_offset
            self._file_objects[block_index].seek(block_offset)
            data = self._file_objects[block_index].read(size)
            buffer.write(data)
            size -= len(data)
            self._offset += len(data)
        return buffer.getvalue()

    def readline(self, size: Optional[int] = None) -> AnyStr:  # pytype: disable=signature-mismatch
        if self._offset >= self._content_size:
            return self._empty_bytes()
        if size is None or size < 0:
            size = self._content_size - self._offset
        block_index, block_offset = self._block_index_and_offset
        self._file_objects[block_index].seek(block_offset)
        data = self._file_objects[block_index].readline(size)
        self._offset += len(data)
        if len(data) == size or (len(data) > 0 and data[-1] == NEWLINE):
            return data
        buffer = self._empty_buffer()
        buffer.write(data)
        while True:
            remain_size = size - buffer.tell()
            block_index, block_offset = self._block_index_and_offset
            self._file_objects[block_index].seek(block_offset)
            data = self._file_objects[block_index].readline(remain_size)
            buffer.write(data)
            self._offset += len(data)
            if buffer.tell() == size or data[-1] == NEWLINE:
                break
        return buffer.getvalue()

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            target_offset = offset
        elif whence == os.SEEK_CUR:
            target_offset = self._offset + offset
        elif whence == os.SEEK_END:
            target_offset = self._content_size + offset
        else:
            raise ValueError("invalid whence: %r" % whence)

        if target_offset < 0:
            raise ValueError("negative seek value %r" % target_offset)

        self._offset = target_offset
        return self._offset

    def _close(self):
        for file_object in self._file_objects:
            file_object.close()
