# Import from alt
from alt.files.http_path import (
    HttpPath,
    get_http_session,
    http_open,
    is_http,
)
from alt.files.interfaces import PathLike, StatResult

__all__ = [
    "get_http_session",
    "is_http",
    "http_open",
    "http_stat",
    "http_getsize",
    "http_getmtime",
    "http_exists",
]


def http_stat(path: PathLike, follow_symlinks=True) -> StatResult:
    """
    Get StatResult of http_url response, including size and mtime, referring to http_getsize and http_getmtime

    :param path: Given path
    :param follow_symlinks: Ignore this parameter, just for compatibility
    :returns: StatResult
    :raises: HttpPermissionError, HttpFileNotFoundError
    """
    return HttpPath(path).stat(follow_symlinks)


def http_getsize(path: PathLike, follow_symlinks: bool = False) -> int:
    """
    Get file size on the given http_url path.

    If http response header don't support Content-Length, will return None

    :param path: Given path
    :param follow_symlinks: Ignore this parameter, just for compatibility
    :returns: File size (in bytes)
    :raises: HttpPermissionError, HttpFileNotFoundError
    """
    return HttpPath(path).getsize(follow_symlinks)


def http_getmtime(path: PathLike, follow_symlinks: bool = False) -> float:
    """
    Get Last-Modified time of the http request on the given http_url path.

    If http response header don't support Last-Modified, will return None

    :param path: Given path
    :param follow_symlinks: Ignore this parameter, just for compatibility
    :returns: Last-Modified time (in Unix timestamp format)
    :raises: HttpPermissionError, HttpFileNotFoundError
    """
    return HttpPath(path).getmtime(follow_symlinks)


def http_exists(path: PathLike, followlinks: bool = False) -> bool:
    """Test if http path exists

    :param path: Given path
    :param followlinks: ignore this parameter, just for compatibility
    :type followlinks: bool, optional
    :return: return True if exists
    :rtype: bool
    """
    return HttpPath(path).exists(followlinks)
