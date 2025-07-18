# Standard Library
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from io import BytesIO
from threading import Lock
from typing import NamedTuple, Optional

# Import from third library
from logging import getLogger as get_logger

# Import from alt
from alt.files.config import (
    BACKOFF_FACTOR,
    BACKOFF_INITIAL,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_MAX_BLOCK_SIZE,
    DEFAULT_MAX_BUFFER_SIZE,
    GLOBAL_MAX_WORKERS,
)
from alt.files.errors import raise_s3_error
from alt.files.interfaces import Writable
from alt.files.utils import get_human_size, process_local

_logger = get_logger(__name__)
"""
class PartResult(NamedTuple):

    etag: str
    part_number: int
    content_size: int

in Python 3.6+
"""

_PartResult = NamedTuple("PartResult", [("etag", str), ("part_number", int), ("content_size", int)])


class PartResult(_PartResult):

    def asdict(self):
        return {
            "PartNumber": self.part_number,
            "ETag": self.etag,
        }


class S3BufferedWriter(Writable):

    def __init__(
        self,
        bucket: str,
        key: str,
        *,
        s3_client,
        block_size: int = DEFAULT_BLOCK_SIZE,
        max_block_size: int = DEFAULT_MAX_BLOCK_SIZE,
        max_buffer_size: int = DEFAULT_MAX_BUFFER_SIZE,
        max_workers: Optional[int] = None,
        profile_name: Optional[str] = None,
    ):

        self._bucket = bucket
        self._key = key
        self._client = s3_client
        self._profile_name = profile_name

        self._block_size = block_size
        self._max_block_size = max_block_size
        self._max_buffer_size = max_buffer_size
        self._total_buffer_size = 0
        self._offset = 0
        self.__content_size = 0
        self._backoff_size = BACKOFF_INITIAL
        self._buffer = BytesIO()

        self._futures = OrderedDict()
        self._is_global_executor = False
        if max_workers is None:
            self._executor = process_local("S3BufferedWriter.executor", ThreadPoolExecutor, max_workers=GLOBAL_MAX_WORKERS)
            self._is_global_executor = True
        else:
            self._executor = ThreadPoolExecutor(max_workers=max_workers)

        self._part_number = 0
        self.__upload_id = None
        self.__upload_id_lock = Lock()

        _logger.debug("open file: %r, mode: %s" % (self.name, self.mode))

    @property
    def name(self) -> str:
        return "s3%s://%s/%s" % (f"+{self._profile_name}" if self._profile_name else "", self._bucket, self._key)

    @property
    def mode(self) -> str:
        return "wb"

    def tell(self) -> int:
        return self._offset

    @property
    def _content_size(self) -> int:
        return self.__content_size

    @_content_size.setter
    def _content_size(self, value: int):
        if value > self._backoff_size:
            _logger.debug("writing file: %r, current size: %s" % (self.name, get_human_size(value)))
        while value > self._backoff_size:
            self._backoff_size *= BACKOFF_FACTOR
        self.__content_size = value

    @property
    def _is_multipart(self) -> bool:
        return len(self._futures) > 0

    @property
    def _upload_id(self) -> str:
        with self.__upload_id_lock:
            if self.__upload_id is None:
                with raise_s3_error(self.name):
                    self.__upload_id = self._client.create_multipart_upload(
                        Bucket=self._bucket,
                        Key=self._key,
                    )["UploadId"]
            return self.__upload_id

    @property
    def _buffer_size(self):
        return self._total_buffer_size - sum(future.result().content_size for future in self._futures.values() if future.done())

    @property
    def _uploading_futures(self):
        return [future for future in self._futures.values() if not future.done()]

    @property
    def _multipart_upload(self):
        return {
            "Parts": [future.result().asdict() for _, future in sorted(self._futures.items())],
        }

    def _upload_buffer(self, part_number, content):
        with raise_s3_error(self.name):
            return PartResult(
                self._client.upload_part(
                    Bucket=self._bucket,
                    Key=self._key,
                    UploadId=self._upload_id,
                    PartNumber=part_number,
                    Body=content,
                )["ETag"],
                part_number,
                len(content),
            )

    def _submit_upload_buffer(self, part_number, content):
        self._futures[part_number] = self._executor.submit(self._upload_buffer, part_number, content)
        self._total_buffer_size += len(content)
        while self._buffer_size > self._max_buffer_size:
            wait(self._uploading_futures, return_when=FIRST_COMPLETED)

    def _submit_upload_content(self, content: bytes):
        # s3 part needs at least 5MB, so we need to divide content into equal-size parts, and give last part more size
        # e.g. 257MB can be divided into 2 parts, 128MB and 129MB
        offset = 0
        while len(content) - offset - self._max_block_size > self._block_size:
            self._part_number += 1
            offset_stop = offset + self._max_block_size
            self._submit_upload_buffer(self._part_number, content[offset:offset_stop])
            offset = offset_stop
        self._part_number += 1
        self._submit_upload_buffer(self._part_number, content[offset:])

    def _submit_futures(self):
        content = self._buffer.getvalue()
        if len(content) == 0:
            return
        self._buffer = BytesIO()
        self._submit_upload_content(content)

    def write(self, data: bytes) -> int:
        if self.closed:
            raise IOError("file already closed: %r" % self.name)

        result = self._buffer.write(data)
        if self._buffer.tell() >= self._block_size:
            self._submit_futures()
        self._offset += result
        self._content_size = self._offset
        return result

    def _shutdown(self):
        if not self._is_global_executor:
            self._executor.shutdown()

    def _close(self):
        _logger.debug("close file: %r" % self.name)

        if not self._is_multipart:
            with raise_s3_error(self.name):
                self._client.put_object(Bucket=self._bucket, Key=self._key, Body=self._buffer.getvalue())
            self._shutdown()
            return

        self._submit_futures()

        with raise_s3_error(self.name):
            self._client.complete_multipart_upload(
                Bucket=self._bucket,
                Key=self._key,
                MultipartUpload=self._multipart_upload,
                UploadId=self._upload_id,
            )

        self._shutdown()
