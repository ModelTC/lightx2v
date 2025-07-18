# Standard Library
from concurrent.futures import Future
from io import BytesIO
from typing import Optional

# Import from alt
from alt.files.config import (
    BACKOFF_FACTOR,
    BACKOFF_INITIAL,
    DEFAULT_BLOCK_CAPACITY,
    DEFAULT_BLOCK_SIZE,
    GLOBAL_MAX_WORKERS,
    NEWLINE,
)
from alt.files.errors import (
    S3FileChangedError,
    S3InvalidRangeError,
    patch_method,
    raise_s3_error,
    s3_should_retry,
)
from alt.files.lib.base_prefetch_reader import (
    BasePrefetchReader,
    LRUCacheFutureManager,
)

__all__ = [
    "DEFAULT_BLOCK_CAPACITY",
    "DEFAULT_BLOCK_SIZE",
    "GLOBAL_MAX_WORKERS",
    "BACKOFF_INITIAL",
    "BACKOFF_FACTOR",
    "NEWLINE",
    "S3PrefetchReader",
    "LRUCacheFutureManager",
]


class S3PrefetchReader(BasePrefetchReader):
    """
    Reader to fast read the s3 content. This will divide the file content into equal parts of block_size size, and will use LRU to cache at most block_capacity blocks in memory.
    open(), seek() and read() will trigger prefetch read. The prefetch will cached block_forward blocks of data from offset position (the position after reading if the called function is read).
    """

    def __init__(
        self,
        bucket: str,
        key: str,
        *,
        s3_client,
        block_size: int = DEFAULT_BLOCK_SIZE,
        block_capacity: int = DEFAULT_BLOCK_CAPACITY,
        block_forward: Optional[int] = None,
        max_retries: int = 10,
        max_workers: Optional[int] = None,
        profile_name: Optional[str] = None,
    ):

        self._bucket = bucket
        self._key = key
        self._client = s3_client
        self._profile_name = profile_name

        super().__init__(
            block_size=block_size,
            block_capacity=block_capacity,
            block_forward=block_forward,
            max_retries=max_retries,
            max_workers=max_workers,
        )

    def _get_content_size(self):
        try:
            start, end = 0, self._block_size - 1
            first_index_response = self._fetch_response(start=start, end=end)
            content_size = int(first_index_response["ContentRange"].split("/")[-1])
        except S3InvalidRangeError:
            # usually when read a empty file
            # TODO: use minio test empty file: https://hub.docker.com/r/minio/minio
            first_index_response = self._fetch_response()
            content_size = int(first_index_response["ContentLength"])

        first_future = Future()
        first_future.set_result(first_index_response["Body"])
        self._insert_futures(index=0, future=first_future)
        self._content_etag = first_index_response["ETag"]
        self._content_info = first_index_response
        return content_size

    @property
    def name(self) -> str:
        return "s3%s://%s/%s" % (f"+{self._profile_name}" if self._profile_name else "", self._bucket, self._key)

    def _fetch_response(self, start: Optional[int] = None, end: Optional[int] = None) -> dict:

        def fetch_response() -> dict:
            if start is None or end is None:
                return self._client.get_object(Bucket=self._bucket, Key=self._key)

            range_str = f"bytes={start}-{end}"
            response = self._client.get_object(Bucket=self._bucket, Key=self._key, Range=range_str)
            response["Body"] = BytesIO(response["Body"].read())
            return response

        fetch_response = patch_method(fetch_response, max_retries=self._max_retries, should_retry=s3_should_retry)

        with raise_s3_error(self.name):
            return fetch_response()

    def _fetch_buffer(self, index: int) -> BytesIO:
        start, end = index * self._block_size, (index + 1) * self._block_size - 1
        response = self._fetch_response(start=start, end=end)
        etag = response.get("ETag", None)
        if etag is not None and etag != self._content_etag:  # pytype: disable=attribute-error
            raise S3FileChangedError(
                "File changed: %r, etag before: %s, after: %s" % (self.name, self._content_info, response)
            )  # pytype: disable=attribute-error

        return response["Body"]
