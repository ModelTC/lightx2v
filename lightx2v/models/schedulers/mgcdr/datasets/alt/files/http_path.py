# Standard Library
import io
import time
from functools import partial
from io import BufferedReader
from typing import Iterable, Iterator, Optional, Tuple, Union

# Import from third library
from logging import getLogger as get_logger

# Import from alt
import requests
from alt.files.config import DEFAULT_BLOCK_SIZE
from alt.files.errors import (
    http_should_retry,
    patch_method,
    translate_http_error,
)
from alt.files.interfaces import PathLike, StatResult, URIPath
from alt.files.lib.compat import fspath
from alt.files.lib.http_prefetch_reader import HttpPrefetchReader
from alt.files.lib.s3_buffered_writer import DEFAULT_MAX_BUFFER_SIZE
from alt.files.lib.url import get_url_scheme
from alt.files.smart_path import SmartPath
from alt.files.utils import binary_open

__all__ = [
    "HttpPath",
    "HttpsPath",
    "get_http_session",
    "is_http",
    "http_open",
]

_logger = get_logger(__name__)
max_retries = 10


def get_http_session(
    timeout: Union[int, Tuple[int, int]] = (9, 60), status_forcelist: Iterable[int] = (500, 502, 503, 504)
) -> requests.Session:
    session = requests.Session()

    def after_callback(response, *args, **kwargs):
        if response.status_code in status_forcelist:
            response.raise_for_status()
        return response

    def before_callback(method, url, **kwargs):
        _logger.debug("send http request: %s %r, with parameters: %s", method, url, kwargs)

    def retry_callback(
        error,
        method,
        url,
        params=None,
        data=None,
        headers=None,
        cookies=None,
        files=None,
        auth=None,
        timeout=None,
        allow_redirects=True,
        proxies=None,
        hooks=None,
        stream=None,
        verify=None,
        cert=None,
        json=None,
        **kwargs,
    ):
        if data and hasattr(data, "seek"):
            data.seek(0)
        elif isinstance(data, Iterator):
            _logger.warning("Can not retry http request with iterator data")
            raise
        if files:

            def seek_or_reopen(file_object):
                if isinstance(file_object, (str, bytes)):
                    return file_object
                elif hasattr(file_object, "seek"):
                    file_object.seek(0)
                    return file_object
                elif hasattr(file_object, "name"):
                    with SmartPath(file_object.name).open("rb") as f:
                        return io.BytesIO(f.read())
                else:
                    _logger.warning('Can not retry http request, because the file object is not seekable and unsupport "name"')
                    raise

            for key, file_info in files.items():
                if hasattr(file_info, "seek"):
                    file_info.seek(0)
                elif isinstance(file_info, (tuple, list)) and len(file_info) >= 2:
                    file_info = list(file_info)
                    if isinstance(file_info[1], (tuple, list)) and len(file_info[1]) >= 2:
                        file_info[1] = list(file_info[1])
                        file_info[1] = seek_or_reopen(file_info[1])
                    else:
                        file_info[1] = seek_or_reopen(file_info[1])
                    files[key] = file_info

    session.request = patch_method(
        partial(session.request, timeout=timeout),
        max_retries=max_retries,
        should_retry=http_should_retry,
        before_callback=before_callback,
        after_callback=after_callback,
        retry_callback=retry_callback,
    )
    return session


def is_http(path: PathLike) -> bool:
    """http scheme definition: http(s)://domain/path

    :param path: Path to be tested
    :returns: True if path is http url, else False
    """

    path = fspath(path)
    if not isinstance(path, str) or not (path.startswith("http://") or path.startswith("https://")):
        return False

    scheme = get_url_scheme(path)
    return scheme == "http" or scheme == "https"


def http_open(
    path: PathLike,
    mode: str = "rb",
    *,
    encoding: Optional[str] = None,
    errors: Optional[str] = None,
    max_concurrency: Optional[int] = None,
    max_buffer_size: int = DEFAULT_MAX_BUFFER_SIZE,
    forward_ratio: Optional[float] = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    **kwargs,
) -> Union[BufferedReader, HttpPrefetchReader]:
    """Open a BytesIO to read binary data of given http(s) url

    .. note ::

        Essentially, it reads data of http(s) url to memory by requests, and then return BytesIO to user.

    :param path: Given path
    :param mode: Only supports 'rb' mode now
    :param encoding: encoding is the name of the encoding used to decode or encode the file. This should only be used in text mode.
    :param errors: errors is an optional string that specifies how encoding and decoding errors are to be handled—this cannot be used in binary mode.
    :param max_concurrency: Max download thread number, None by default
    :param max_buffer_size: Max cached buffer size in memory, 128MB by default
    :param block_size: Size of single block, 8MB by default. Each block will be uploaded or downloaded by single thread.
    :return: BytesIO initialized with http(s) data
    """
    return HttpPath(path).open(
        mode,
        encoding=encoding,
        errors=errors,
        max_concurrency=max_concurrency,
        max_buffer_size=max_buffer_size,
        forward_ratio=forward_ratio,
        block_size=block_size,
    )


@SmartPath.register
class HttpPath(URIPath):

    protocol = "http"

    def __init__(self, path: PathLike, *other_paths: PathLike):
        if str(path).startswith("https://"):
            self.protocol = "https"
        super().__init__(path, *other_paths)

    @binary_open
    def open(
        self,
        mode: str = "rb",
        *,
        max_concurrency: Optional[int] = None,
        max_buffer_size: int = DEFAULT_MAX_BUFFER_SIZE,
        forward_ratio: Optional[float] = None,
        block_size: int = DEFAULT_BLOCK_SIZE,
        **kwargs,
    ) -> Union[BufferedReader, HttpPrefetchReader]:
        """Open a BytesIO to read binary data of given http(s) url

        .. note ::

            Essentially, it reads data of http(s) url to memory by requests, and then return BytesIO to user.

        :param mode: Only supports 'rb' mode now
        :param encoding: encoding is the name of the encoding used to decode or encode the file. This should only be used in text mode.
        :param errors: errors is an optional string that specifies how encoding and decoding errors are to be handled—this cannot be used in binary mode.
        :param max_concurrency: Max download thread number, None by default
        :param max_buffer_size: Max cached buffer size in memory, 128MB by default
        :param block_size: Size of single block, 8MB by default. Each block will be uploaded or downloaded by single thread.
        :return: BytesIO initialized with http(s) data
        """
        if mode not in ("rb",):
            raise ValueError("unacceptable mode: %r" % mode)

        response = None
        try:
            response = get_http_session(status_forcelist=()).get(self.path_with_protocol, stream=True)
            response.raise_for_status()
        except Exception as error:
            if response:
                response.close()
            raise translate_http_error(error, self.path_with_protocol)

        content_size = int(response.headers["Content-Length"])
        if response.headers.get("Accept-Ranges") == "bytes" and content_size >= block_size * 2:
            response.close()

            block_capacity = max_buffer_size // block_size
            if forward_ratio is None:
                block_forward = None
            else:
                block_forward = max(int(block_capacity * forward_ratio), 1)

            return HttpPrefetchReader(
                self.path_with_protocol,
                content_size=content_size,
                max_retries=max_retries,
                max_workers=max_concurrency,
                block_capacity=block_capacity,
                block_forward=block_forward,
                block_size=block_size,
            )
        response.raw.auto_close = False
        response.raw.name = self.path_with_protocol
        return BufferedReader(response.raw)

    def stat(self, follow_symlinks=True) -> StatResult:
        """
        Get StatResult of http_url response, including size and mtime, referring to http_getsize and http_getmtime

        :param follow_symlinks: Ignore this parameter, just for compatibility
        :returns: StatResult
        :raises: HttpPermissionError, HttpFileNotFoundError
        """

        try:
            with get_http_session(status_forcelist=()).get(self.path_with_protocol, stream=True) as response:
                response.raise_for_status()
                headers = response.headers
        except Exception as error:
            raise translate_http_error(error, self.path_with_protocol)

        size = headers.get("Content-Length")
        if size:
            size = int(size)

        last_modified = headers.get("Last-Modified")
        if last_modified:
            last_modified = time.mktime(time.strptime(last_modified, "%a, %d %b %Y %H:%M:%S %Z"))

        return StatResult(size=size, mtime=last_modified, isdir=False, islnk=False, extra=headers)  # pyre-ignore[20]

    def getsize(self, follow_symlinks: bool = False) -> int:
        """
        Get file size on the given http_url path.

        If http response header don't support Content-Length, will return None

        :param follow_symlinks: Ignore this parameter, just for compatibility
        :returns: File size (in bytes)
        :raises: HttpPermissionError, HttpFileNotFoundError
        """
        return self.stat().size

    def getmtime(self, follow_symlinks: bool = False) -> float:
        """
        Get Last-Modified time of the http request on the given http_url path.

        If http response header don't support Last-Modified, will return None

        :param follow_symlinks: Ignore this parameter, just for compatibility
        :returns: Last-Modified time (in Unix timestamp format)
        :raises: HttpPermissionError, HttpFileNotFoundError
        """
        return self.stat().mtime

    def exists(self, followlinks: bool = False) -> bool:
        """Test if http path exists

        :param followlinks: ignore this parameter, just for compatibility
        :type followlinks: bool, optional
        :return: return True if exists
        :rtype: bool
        """
        try:
            with get_http_session(status_forcelist=()).get(self.path_with_protocol, stream=True) as response:
                if response.status_code == 404:
                    return False
                return True
        except requests.exceptions.ConnectionError:
            return False


@SmartPath.register
class HttpsPath(HttpPath):

    protocol = "https"
