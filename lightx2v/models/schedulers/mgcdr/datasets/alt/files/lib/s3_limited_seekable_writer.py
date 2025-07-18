# Standard Library
import os
from io import BytesIO
from typing import Optional

# Import from third library
from logging import getLogger as get_logger

# Import from alt
from alt.files.config import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_MAX_BLOCK_SIZE,
    DEFAULT_MAX_BUFFER_SIZE,
)
from alt.files.errors import raise_s3_error
from alt.files.interfaces import Seekable
from alt.files.lib.s3_buffered_writer import S3BufferedWriter

_logger = get_logger(__name__)


class S3LimitedSeekableWriter(Seekable, S3BufferedWriter):
    """For file format like msgpack and mp4, it's a pain that you need to write
    header before writing the data. So it's kind of hard to make streaming write
    to unseekable file system like s3. In this case, we will try to keep the first
    and last parts of data in memory, so we can come back to head again and write
    the header at the last second.
    """

    def __init__(
        self,
        bucket: str,
        key: str,
        *,
        s3_client,
        block_size: int = DEFAULT_BLOCK_SIZE,
        head_block_size: Optional[int] = None,
        tail_block_size: Optional[int] = None,
        max_block_size: int = DEFAULT_MAX_BLOCK_SIZE,
        max_buffer_size: int = DEFAULT_MAX_BUFFER_SIZE,
        max_workers: Optional[int] = None,
        profile_name: Optional[str] = None
    ):

        super().__init__(
            bucket,
            key,
            s3_client=s3_client,
            block_size=block_size,
            max_block_size=max_block_size,
            max_buffer_size=max_buffer_size,
            max_workers=max_workers,
            profile_name=profile_name,
        )

        self._head_block_size = head_block_size or block_size
        self._tail_block_size = tail_block_size or block_size
        self._head_buffer = BytesIO()

    @property
    def _head_size(self) -> int:
        return len(self._head_buffer.getvalue())

    @property
    def _tail_size(self) -> int:
        return len(self._buffer.getvalue())

    @property
    def _tail_offset(self) -> int:
        return self._content_size - self._tail_size

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if self.closed:
            raise IOError("file already closed: %r" % self.name)

        if whence == os.SEEK_SET:
            target_offset = offset
        elif whence == os.SEEK_CUR:
            target_offset = self._offset + offset
        elif whence == os.SEEK_END:
            target_offset = self._content_size + offset

        if target_offset < self._head_block_size:
            self._head_buffer.seek(target_offset)
        elif target_offset >= self._tail_offset:
            self._buffer.seek(target_offset - self._tail_offset)
        else:
            raise OSError("Can only seek inside of head, or seek to tail, target offset: %d" % target_offset)

        self._offset = target_offset
        return self._offset

    def write(self, data: bytes) -> int:
        if self.closed:
            raise IOError("file already closed: %r" % self.name)

        if self._head_size != self._head_block_size:  # no tail part yet
            self._write_to_head(data)
        elif self._offset < self._head_block_size:  # tail part already created
            self._write_to_head_after_tail_part_created(data)
        elif self._offset >= self._tail_offset:
            self._write_to_tail(data)
        else:
            raise OSError("Can only write inside of head, or write to tail, current offset: %d" % self._offset)
        return len(data)

    def _write_to_head(self, data: bytes):
        if self._offset + len(data) <= self._head_block_size:
            self._head_buffer.write(data)
            self._content_size = self._offset = self._head_size
            if self._content_size == self._head_block_size:
                self._part_number += 1
        else:  # head part exceeded
            offset = self._head_block_size - self._offset
            self._head_buffer.write(data[:offset])
            self._content_size = self._offset = self._head_size
            self._part_number += 1
            self._write_to_tail(data[offset:])

    def _write_to_head_after_tail_part_created(self, data: bytes):
        if self._offset + len(data) > self._head_block_size:
            raise Exception(
                "Head part overflow, %d bytes left but try to write %d bytes"
                % (self._head_block_size - self._offset, len(data))
            )
        self._head_buffer.write(data)
        self._offset += len(data)

    def _write_to_tail(self, data: bytes):
        self._buffer.write(data)
        if self._buffer.tell() >= self._block_size + self._tail_block_size:
            self._submit_futures()
        self._offset += len(data)
        if self._offset > self._content_size:
            self._content_size = self._offset

    def _submit_futures(self):
        content = self._buffer.getvalue()
        if len(content) == 0:
            return
        offset = len(content) - self._tail_block_size
        self._buffer = BytesIO(content[offset:])
        self._buffer.seek(0, os.SEEK_END)
        self._submit_upload_content(content[:offset])

    def _close(self):
        _logger.debug("close file: %r" % self.name)

        if not self._is_multipart:
            with raise_s3_error(self.name):
                self._client.put_object(
                    Bucket=self._bucket, Key=self._key, Body=self._head_buffer.getvalue() + self._buffer.getvalue()
                )
            self._shutdown()
            return

        self._submit_upload_buffer(1, self._head_buffer.getvalue())
        self._head_buffer = BytesIO()  # clean memory

        content = self._buffer.getvalue()
        if len(content) > 0:
            self._submit_upload_content(content)
        self._buffer = BytesIO()  # clean memory

        with raise_s3_error(self.name):
            self._client.complete_multipart_upload(
                Bucket=self._bucket,
                Key=self._key,
                MultipartUpload=self._multipart_upload,
                UploadId=self._upload_id,
            )

        self._shutdown()
