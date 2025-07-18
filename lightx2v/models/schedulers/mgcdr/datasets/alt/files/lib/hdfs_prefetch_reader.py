# Standard Library
from io import BytesIO
from typing import Optional

# Import from alt
from alt.files.config import DEFAULT_BLOCK_CAPACITY, DEFAULT_BLOCK_SIZE
from alt.files.errors import raise_hdfs_error
from alt.files.lib.base_prefetch_reader import BasePrefetchReader


class HdfsPrefetchReader(BasePrefetchReader):
    """
    Reader to fast read the hdfs content. This will divide the file content into equal parts of block_size size, and will use LRU to cache at most block_capacity blocks in memory.
    open(), seek() and read() will trigger prefetch read. The prefetch will cached block_forward blocks of data from offset position (the position after reading if the called function is read).
    """

    def __init__(
        self,
        hdfs_path: str,
        *,
        client,
        block_size: int = DEFAULT_BLOCK_SIZE,
        block_capacity: int = DEFAULT_BLOCK_CAPACITY,
        block_forward: Optional[int] = None,
        max_retries: int = 10,
        max_workers: Optional[int] = None,
        profile_name: Optional[str] = None,
    ):
        self._path = hdfs_path
        self._client = client
        self._profile_name = profile_name

        super().__init__(
            block_size=block_size,
            block_capacity=block_capacity,
            block_forward=block_forward,
            max_retries=max_retries,
            max_workers=max_workers,
        )

    def _get_content_size(self):
        with raise_hdfs_error(self._path):
            return self._client.status(self._path)["length"]

    @property
    def name(self) -> str:
        return "hdfs%s://%s" % (f"+{self._profile_name}" if self._profile_name else "", self._path)

    def _fetch_response(self, start: Optional[int] = None, end: Optional[int] = None) -> dict:
        with raise_hdfs_error(self.name):
            with self._client.read(self._path, offset=start or 0, length=end - start if start and end else None) as f:
                return {"Body": BytesIO(f.read())}
