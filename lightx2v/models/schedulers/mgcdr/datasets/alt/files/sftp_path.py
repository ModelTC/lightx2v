# Standard Library
import atexit
import fcntl
import hashlib
import io
import os
import random
import shlex
import socket
import subprocess
from stat import S_ISDIR, S_ISLNK, S_ISREG
from typing import (
    IO,
    AnyStr,
    BinaryIO,
    Callable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)
from urllib.parse import urlsplit, urlunsplit

# Import from third library
from logging import getLogger as get_logger

# Import from alt
import paramiko
from alt.files.errors import (
    SameFileError,
    _create_missing_ok_generator,
    patch_method,
)
from alt.files.interfaces import (
    ContextIterator,
    FileEntry,
    PathLike,
    StatResult,
)
from alt.files.lib.compare import is_same_file
from alt.files.lib.compat import fspath
from alt.files.lib.glob import FSFunc, iglob
from alt.files.lib.joinpath import uri_join
from alt.files.pathlike import URIPath
from alt.files.smart_path import SmartPath
from alt.files.utils import cachedproperty, calculate_md5, thread_local

_logger = get_logger(__name__)

__all__ = [
    "SftpPath",
    "is_sftp",
    "sftp_readlink",
    "sftp_glob",
    "sftp_iglob",
    "sftp_glob_stat",
    "sftp_resolve",
    "sftp_download",
    "sftp_upload",
    "sftp_path_join",
    "sftp_concat",
    "sftp_lstat",
]

SFTP_USERNAME = "SFTP_USERNAME"
SFTP_PASSWORD = "SFTP_PASSWORD"
SFTP_PRIVATE_KEY_PATH = "SFTP_PRIVATE_KEY_PATH"
SFTP_PRIVATE_KEY_TYPE = "SFTP_PRIVATE_KEY_TYPE"
SFTP_PRIVATE_KEY_PASSWORD = "SFTP_PRIVATE_KEY_PASSWORD"
SFTP_MAX_UNAUTH_CONN = "SFTP_MAX_UNAUTH_CONN"
MAX_RETRIES = 10
DEFAULT_SSH_CONNECT_TIMEOUT = 5
DEFAULT_SSH_KEEPALIVE_INTERVAL = 15


def _make_stat(stat: paramiko.SFTPAttributes) -> StatResult:
    return StatResult(
        size=stat.st_size,
        mtime=stat.st_mtime,
        isdir=S_ISDIR(stat.st_mode),
        islnk=S_ISLNK(stat.st_mode),
        extra=stat,
    )


def get_private_key():
    key_with_types = {
        "DSA": paramiko.DSSKey,
        "RSA": paramiko.RSAKey,
        "ECDSA": paramiko.ECDSAKey,
        "ED25519": paramiko.Ed25519Key,
    }
    key_type = os.getenv(SFTP_PRIVATE_KEY_TYPE, "RSA").upper()
    if os.getenv(SFTP_PRIVATE_KEY_PATH):
        private_key_path = os.getenv(SFTP_PRIVATE_KEY_PATH)
        if not os.path.exists(private_key_path):
            raise FileNotFoundError(f"Private key file not exist: '{SFTP_PRIVATE_KEY_PATH}'")
        return key_with_types[key_type].from_private_key_file(private_key_path, password=os.getenv(SFTP_PRIVATE_KEY_PASSWORD))
    return None


def provide_connect_info(
    hostname: str,
    port: Optional[int] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
):
    if not port:
        port = 22
    if not username:
        username = os.getenv(SFTP_USERNAME)
    if not password:
        password = os.getenv(SFTP_PASSWORD)
    private_key = get_private_key()
    return hostname, port, username, password, private_key


def sftp_should_retry(error: Exception) -> bool:
    if type(error) is EOFError:
        return False
    elif isinstance(
        error,
        (
            paramiko.ssh_exception.SSHException,
            ConnectionError,
            socket.timeout,
        ),
    ):
        return True
    elif isinstance(error, OSError):
        for err_msg in [
            "Socket is closed",
            "Cannot assign requested address",
        ]:
            if err_msg in str(error):
                return True
    return False


def _patch_sftp_client_request(
    client: paramiko.SFTPClient,
    hostname: str,
    port: Optional[int] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
):

    def retry_callback(error, *args, **kwargs):
        client.close()
        ssh_client = get_ssh_client(hostname, port, username, password)
        ssh_client.close()
        atexit.unregister(ssh_client.close)
        ssh_key = f"ssh_client:{hostname},{port},{username},{password}"
        if thread_local.get(ssh_key):
            del thread_local[ssh_key]
        sftp_key = f"sftp_client:{hostname},{port},{username},{password}"
        if thread_local.get(sftp_key):
            del thread_local[sftp_key]

        new_sftp_client = get_sftp_client(
            hostname=hostname,
            port=port,
            username=username,
            password=password,
        )
        client.sock = new_sftp_client.sock

    client._request = patch_method(
        client._request,  # pytype: disable=attribute-error
        max_retries=MAX_RETRIES,
        should_retry=sftp_should_retry,
        retry_callback=retry_callback,
    )
    return client


def _get_sftp_client(
    hostname: str,
    port: Optional[int] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> paramiko.SFTPClient:
    """Get sftp client

    :returns: sftp client
    """
    session = get_ssh_session(
        hostname=hostname,
        port=port,
        username=username,
        password=password,
    )
    session.invoke_subsystem("sftp")
    sftp_client = paramiko.SFTPClient(session)
    _patch_sftp_client_request(sftp_client, hostname, port, username, password)
    return sftp_client


def get_sftp_client(
    hostname: str,
    port: Optional[int] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> paramiko.SFTPClient:
    """Get sftp client

    :returns: sftp client
    """
    return thread_local(
        f"sftp_client:{hostname},{port},{username},{password}", _get_sftp_client, hostname, port, username, password
    )


def _get_ssh_client(
    hostname: str,
    port: Optional[int] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> paramiko.SSHClient:
    hostname, port, username, password, private_key = provide_connect_info(
        hostname=hostname,
        port=port,
        username=username,
        password=password,
    )

    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    max_unauth_connections = int(os.getenv(SFTP_MAX_UNAUTH_CONN, 10))
    try:
        fd = os.open(
            os.path.join("/tmp", f"alt.file-sftp-{hostname}-{random.randint(1, max_unauth_connections)}"),
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        )
    except Exception:
        _logger.warning("Can't create file lock in '/tmp', please control the SFTP concurrency count by yourself.")
        fd = None
    if fd:
        fcntl.flock(fd, fcntl.LOCK_EX)
    ssh_client.connect(
        hostname=hostname,
        port=port,
        username=username,
        password=password,
        pkey=private_key,
        timeout=DEFAULT_SSH_CONNECT_TIMEOUT,
        auth_timeout=DEFAULT_SSH_CONNECT_TIMEOUT,
        banner_timeout=DEFAULT_SSH_CONNECT_TIMEOUT,
    )
    if fd:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    atexit.register(ssh_client.close)
    return ssh_client


def get_ssh_client(
    hostname: str,
    port: Optional[int] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> paramiko.SSHClient:
    return thread_local(
        f"ssh_client:{hostname},{port},{username},{password}", _get_ssh_client, hostname, port, username, password
    )


def get_ssh_session(
    hostname: str,
    port: Optional[int] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> paramiko.Channel:

    def retry_callback(error, *args, **kwargs):
        ssh_client = get_ssh_client(hostname, port, username, password)
        ssh_client.close()
        atexit.unregister(ssh_client.close)
        ssh_key = f"ssh_client:{hostname},{port},{username},{password}"
        if thread_local.get(ssh_key):
            del thread_local[ssh_key]
        sftp_key = f"sftp_client:{hostname},{port},{username},{password}"
        if thread_local.get(sftp_key):
            del thread_local[sftp_key]

    return patch_method(
        _open_session,  # pytype: disable=attribute-error
        max_retries=MAX_RETRIES,
        should_retry=sftp_should_retry,
        retry_callback=retry_callback,
    )(
        hostname,
        port,
        username,
        password,
    )


def _open_session(
    hostname: str,
    port: Optional[int] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> paramiko.Channel:
    ssh_client = get_ssh_client(hostname, port, username, password)
    transport = ssh_client.get_transport()
    if not transport:
        raise paramiko.SSHException("Get transport error")
    transport.set_keepalive(DEFAULT_SSH_KEEPALIVE_INTERVAL)
    session = transport.open_session(timeout=DEFAULT_SSH_CONNECT_TIMEOUT)
    if not session:
        raise paramiko.SSHException("Create session error")
    session.settimeout(DEFAULT_SSH_CONNECT_TIMEOUT)
    return session


def is_sftp(path: PathLike) -> bool:
    """Test if a path is sftp path

    :param path: Path to be tested
    :returns: True of a path is sftp path, else False
    """
    path = fspath(path)
    parts = urlsplit(path)
    return parts.scheme == "sftp"


def sftp_readlink(path: PathLike) -> "str":
    """
    Return a SftpPath instance representing the path to which the symbolic link points.
    :param path: Given path
    :returns: Return a SftpPath instance representing the path to which the symbolic link points.
    """
    return SftpPath(path).readlink().path_with_protocol


def sftp_glob(path: PathLike, recursive: bool = True, missing_ok: bool = True) -> List[str]:
    """Return path list in ascending alphabetical order, in which path matches glob pattern

    1. If doesn't match any path, return empty list
        Notice:  ``glob.glob`` in standard library returns ['a/'] instead of empty list when pathname is like `a/**`, recursive is True and directory 'a' doesn't exist. fs_glob behaves like ``glob.glob`` in standard library under such circumstance.
    2. No guarantee that each path in result is different, which means:
        Assume there exists a path `/a/b/c/b/d.tmp.txt`
        use path pattern like `/**/b/**/*.tmp.txt` to glob, the path above will be returned twice
    3. `**` will match any matched file, directory, symlink and '' by default, when recursive is `True`
    4. fs_glob returns same as glob.glob(pathname, recursive=True) in acsending alphabetical order.
    5. Hidden files (filename stars with '.') will not be found in the result

    :param path: Given path
    :param pattern: Glob the given relative pattern in the directory represented by this path
    :param recursive: If False, `**` will not search directory recursively
    :param missing_ok: If False and target path doesn't match any file, raise FileNotFoundError
    :returns: A list contains paths match `pathname`
    """
    return list(sftp_iglob(path=path, recursive=recursive, missing_ok=missing_ok))


def sftp_glob_stat(path: PathLike, recursive: bool = True, missing_ok: bool = True) -> Iterator[FileEntry]:
    """Return a list contains tuples of path and file stat, in ascending alphabetical order, in which path matches glob pattern

    1. If doesn't match any path, return empty list
        Notice:  ``glob.glob`` in standard library returns ['a/'] instead of empty list when pathname is like `a/**`, recursive is True and directory 'a' doesn't exist. sftp_glob behaves like ``glob.glob`` in standard library under such circumstance.
    2. No guarantee that each path in result is different, which means:
        Assume there exists a path `/a/b/c/b/d.tmp.txt`
        use path pattern like `/**/b/**/*.tmp.txt` to glob, the path above will be returned twice
    3. `**` will match any matched file, directory, symlink and '' by default, when recursive is `True`
    4. fs_glob returns same as glob.glob(pathname, recursive=True) in acsending alphabetical order.
    5. Hidden files (filename stars with '.') will not be found in the result

    :param path: Given path
    :param pattern: Glob the given relative pattern in the directory represented by this path
    :param recursive: If False, `**` will not search directory recursively
    :param missing_ok: If False and target path doesn't match any file, raise FileNotFoundError
    :returns: A list contains tuples of path and file stat, in which paths match `pathname`
    """
    for path in sftp_iglob(path=path, recursive=recursive, missing_ok=missing_ok):
        path_object = SftpPath(path)
        yield FileEntry(path_object.name, path_object.path_with_protocol, path_object.lstat())


def sftp_iglob(path: PathLike, recursive: bool = True, missing_ok: bool = True) -> Iterator[str]:
    """Return path iterator in ascending alphabetical order, in which path matches glob pattern

    1. If doesn't match any path, return empty list
        Notice:  ``glob.glob`` in standard library returns ['a/'] instead of empty list when pathname is like `a/**`, recursive is True and directory 'a' doesn't exist. fs_glob behaves like ``glob.glob`` in standard library under such circumstance.
    2. No guarantee that each path in result is different, which means:
        Assume there exists a path `/a/b/c/b/d.tmp.txt`
        use path pattern like `/**/b/**/*.tmp.txt` to glob, the path above will be returned twice
    3. `**` will match any matched file, directory, symlink and '' by default, when recursive is `True`
    4. fs_glob returns same as glob.glob(pathname, recursive=True) in acsending alphabetical order.
    5. Hidden files (filename stars with '.') will not be found in the result

    :param path: Given path
    :param pattern: Glob the given relative pattern in the directory represented by this path
    :param recursive: If False, `**` will not search directory recursively
    :param missing_ok: If False and target path doesn't match any file, raise FileNotFoundError
    :returns: An iterator contains paths match `pathname`
    """

    for path in SftpPath(path).iglob(pattern="", recursive=recursive, missing_ok=missing_ok):
        yield path.path_with_protocol


def sftp_resolve(path: PathLike, strict=False) -> "str":
    """Equal to fs_realpath

    :param path: Given path
    :param strict: Ignore this parameter, just for compatibility
    :return: Return the canonical path of the specified filename, eliminating any symbolic links encountered in the path.
    :rtype: SftpPath
    """
    return SftpPath(path).resolve(strict).path_with_protocol


def _sftp_scan_pairs(src_url: PathLike, dst_url: PathLike) -> Iterator[Tuple[PathLike, PathLike]]:
    for src_file_path in SftpPath(src_url).scan():
        content_path = src_file_path[len(src_url) :]
        if len(content_path) > 0:
            dst_file_path = SftpPath(dst_url).joinpath(content_path).path_with_protocol
        else:
            dst_file_path = dst_url
        yield src_file_path, dst_file_path


def sftp_download(
    src_url: PathLike, dst_url: PathLike, callback: Optional[Callable[[int], None]] = None, followlinks: bool = False
):
    """
    File download
    """
    # Import from alt
    from alt.files.fs import is_fs
    from alt.files.fs_path import FSPath

    if not is_fs(dst_url):
        raise OSError(f"dst_url is not fs path: {dst_url}")
    if not is_sftp(src_url) and not isinstance(src_url, SftpPath):
        raise OSError(f"src_url is not sftp path: {src_url}")

    if isinstance(src_url, SftpPath):
        src_path = src_url
    else:
        src_path = SftpPath(src_url)

    if followlinks and src_path.is_symlink():
        src_path = src_path.readlink()
    if src_path.is_dir():
        raise IsADirectoryError("Is a directory: %r" % src_url)
    if str(dst_url).endswith("/"):
        raise IsADirectoryError("Is a directory: %r" % dst_url)

    dst_path = FSPath(dst_url)
    dst_path.parent.makedirs(exist_ok=True)

    sftp_callback = None
    if callback:
        bytes_transferred_before = 0

        def sftp_callback(bytes_transferred: int, _total_bytes: int):  # noqa
            nonlocal bytes_transferred_before
            callback(bytes_transferred - bytes_transferred_before)
            bytes_transferred_before = bytes_transferred

    src_path._client.get(src_path._real_path, dst_path.path_without_protocol, callback=sftp_callback)

    src_stat = src_path.stat()
    dst_path.utime(src_stat.st_atime, src_stat.st_mtime)
    dst_path.chmod(src_stat.st_mode)


def sftp_upload(
    src_url: PathLike, dst_url: PathLike, callback: Optional[Callable[[int], None]] = None, followlinks: bool = False
):
    """
    File upload
    """
    # Import from alt
    from alt.files.fs import is_fs
    from alt.files.fs_path import FSPath

    if not is_fs(src_url):
        raise OSError(f"src_url is not fs path: {src_url}")
    if not is_sftp(dst_url) and not isinstance(dst_url, SftpPath):
        raise OSError(f"dst_url is not sftp path: {dst_url}")

    if followlinks and os.path.islink(src_url):
        src_url = os.readlink(src_url)
    if os.path.isdir(src_url):
        raise IsADirectoryError("Is a directory: %r" % src_url)
    if str(dst_url).endswith("/"):
        raise IsADirectoryError("Is a directory: %r" % dst_url)

    src_path = FSPath(src_url)
    if isinstance(dst_url, SftpPath):
        dst_path = dst_url
    else:
        dst_path = SftpPath(dst_url)
    dst_path.parent.makedirs(exist_ok=True)

    sftp_callback = None
    if callback:
        bytes_transferred_before = 0

        def sftp_callback(bytes_transferred: int, _total_bytes: int):  # noqa
            nonlocal bytes_transferred_before
            callback(bytes_transferred - bytes_transferred_before)
            bytes_transferred_before = bytes_transferred

    dst_path._client.put(src_path.path_without_protocol, dst_path._real_path, callback=sftp_callback)

    src_stat = src_path.stat()
    dst_path.utime(src_stat.st_atime, src_stat.st_mtime)
    dst_path.chmod(src_stat.st_mode)


def sftp_path_join(path: PathLike, *other_paths: PathLike) -> str:
    """
    Concat 2 or more path to a complete path

    :param path: Given path
    :param other_paths: Paths to be concatenated
    :returns: Concatenated complete path

    .. note ::

        The difference between this function and ``os.path.join`` is that this function ignores left side slash (which indicates absolute path) in ``other_paths`` and will directly concat.
        e.g. os.path.join('/path', 'to', '/file') => '/file', but sftp_path_join('/path', 'to', '/file') => '/path/to/file'
    """
    return uri_join(fspath(path), *map(fspath, other_paths))


def sftp_concat(src_paths: List[PathLike], dst_path: PathLike) -> None:
    """Concatenate sftp files to one file.

    :param src_paths: Given source paths
    :param dst_path: Given destination path
    """
    dst_path_obj = SftpPath(dst_path)

    def get_real_path(path: PathLike) -> str:
        return SftpPath(path)._real_path

    command = ["cat", *map(get_real_path, src_paths), ">", get_real_path(dst_path)]
    exec_result = dst_path_obj._exec_command(command)
    if exec_result.returncode != 0:
        _logger.error(exec_result.stderr)
        raise OSError(f"Failed to concat {src_paths} to {dst_path}")


def sftp_lstat(path: PathLike) -> StatResult:
    """
    Get StatResult of file on sftp, including file size and mtime, referring to fs_getsize and fs_getmtime

    :param path: Given path
    :returns: StatResult
    """
    return SftpPath(path).lstat()


@SmartPath.register
class SftpPath(URIPath):
    """sftp protocol

    uri format:
    - absolute path
        - sftp://[username[:password]@]hostname[:port]//file_path
    - relative path
        - - sftp://[username[:password]@]hostname[:port]/file_path
    """

    protocol = "sftp"

    def __init__(self, path: "PathLike", *other_paths: "PathLike"):
        super().__init__(path, *other_paths)
        parts = urlsplit(self.path)
        self._urlsplit_parts = parts
        self._real_path = parts.path
        if parts.path.startswith("//"):
            self._root_dir = "/"
        else:
            self._root_dir = self._client.normalize(".")
        self._real_path = os.path.join(self._root_dir, parts.path.lstrip("/"))

    @cachedproperty
    def parts(self) -> Tuple[str]:
        """A tuple giving access to the path’s various components"""
        if self._urlsplit_parts.path.startswith("//"):
            new_parts = self._urlsplit_parts._replace(path="//")
        else:
            new_parts = self._urlsplit_parts._replace(path="/")
        parts = [urlunsplit(new_parts)]
        path = self._urlsplit_parts.path.lstrip("/")
        if path != "":
            parts.extend(path.split("/"))
        return tuple(parts)

    @property
    def _client(self):
        return get_sftp_client(
            hostname=self._urlsplit_parts.hostname,
            port=self._urlsplit_parts.port,
            username=self._urlsplit_parts.username,
            password=self._urlsplit_parts.password,
        )

    def _generate_path_object(self, sftp_local_path: str, resolve: bool = False):
        if resolve or self._root_dir == "/":
            sftp_local_path = f"//{sftp_local_path.lstrip('/')}"
        else:
            sftp_local_path = os.path.relpath(sftp_local_path, start=self._root_dir)
            if sftp_local_path == ".":
                sftp_local_path = "/"
        new_parts = self._urlsplit_parts._replace(path=sftp_local_path)
        return self.from_path(urlunsplit(new_parts))

    def exists(self, followlinks: bool = False) -> bool:
        """
        Test if the path exists

        :param followlinks: False if regard symlink as file, else True
        :returns: True if the path exists, else False

        """
        try:
            if followlinks:
                self._client.stat(self._real_path)
            else:
                self._client.lstat(self._real_path)
            return True
        except FileNotFoundError:
            return False

    def getmtime(self, follow_symlinks: bool = False) -> float:
        """
        Get last-modified time of the file on the given path (in Unix timestamp format).
        If the path is an existent directory, return the latest modified time of all file in it.

        :returns: last-modified time
        """
        return self.stat(follow_symlinks=follow_symlinks).mtime

    def getsize(self, follow_symlinks: bool = False) -> int:
        """
        Get file size on the given file path (in bytes).
        If the path in a directory, return the sum of all file size in it, including file in subdirectories (if exist).
        The result excludes the size of directory itself. In other words, return 0 Byte on an empty directory path.

        :returns: File size

        """
        return self.stat(follow_symlinks=follow_symlinks).size

    def glob(self, pattern, recursive: bool = True, missing_ok: bool = True) -> List["SftpPath"]:
        """Return path list in ascending alphabetical order, in which path matches glob pattern

        1. If doesn't match any path, return empty list
            Notice:  ``glob.glob`` in standard library returns ['a/'] instead of empty list when pathname is like `a/**`, recursive is True and directory 'a' doesn't exist. fs_glob behaves like ``glob.glob`` in standard library under such circumstance.
        2. No guarantee that each path in result is different, which means:
            Assume there exists a path `/a/b/c/b/d.tmp.txt`
            use path pattern like `/**/b/**/*.tmp.txt` to glob, the path above will be returned twice
        3. `**` will match any matched file, directory, symlink and '' by default, when recursive is `True`
        4. fs_glob returns same as glob.glob(pathname, recursive=True) in acsending alphabetical order.
        5. Hidden files (filename stars with '.') will not be found in the result

        :param pattern: Glob the given relative pattern in the directory represented by this path
        :param recursive: If False, `**` will not search directory recursively
        :param missing_ok: If False and target path doesn't match any file, raise FileNotFoundError
        :returns: A list contains paths match `pathname`
        """
        return list(self.iglob(pattern=pattern, recursive=recursive, missing_ok=missing_ok))

    def glob_stat(self, pattern, recursive: bool = True, missing_ok: bool = True) -> Iterator[FileEntry]:
        """Return a list contains tuples of path and file stat, in ascending alphabetical order, in which path matches glob pattern

        1. If doesn't match any path, return empty list
            Notice:  ``glob.glob`` in standard library returns ['a/'] instead of empty list when pathname is like `a/**`, recursive is True and directory 'a' doesn't exist. sftp_glob behaves like ``glob.glob`` in standard library under such circumstance.
        2. No guarantee that each path in result is different, which means:
            Assume there exists a path `/a/b/c/b/d.tmp.txt`
            use path pattern like `/**/b/**/*.tmp.txt` to glob, the path above will be returned twice
        3. `**` will match any matched file, directory, symlink and '' by default, when recursive is `True`
        4. fs_glob returns same as glob.glob(pathname, recursive=True) in acsending alphabetical order.
        5. Hidden files (filename stars with '.') will not be found in the result

        :param pattern: Glob the given relative pattern in the directory represented by this path
        :param recursive: If False, `**` will not search directory recursively
        :param missing_ok: If False and target path doesn't match any file, raise FileNotFoundError
        :returns: A list contains tuples of path and file stat, in which paths match `pathname`
        """
        for path_obj in self.iglob(pattern=pattern, recursive=recursive, missing_ok=missing_ok):
            yield FileEntry(path_obj.name, path_obj.path, path_obj.lstat())

    def iglob(self, pattern, recursive: bool = True, missing_ok: bool = True) -> Iterator["SftpPath"]:
        """Return path iterator in ascending alphabetical order, in which path matches glob pattern

        1. If doesn't match any path, return empty list
            Notice:  ``glob.glob`` in standard library returns ['a/'] instead of empty list when pathname is like `a/**`, recursive is True and directory 'a' doesn't exist. fs_glob behaves like ``glob.glob`` in standard library under such circumstance.
        2. No guarantee that each path in result is different, which means:
            Assume there exists a path `/a/b/c/b/d.tmp.txt`
            use path pattern like `/**/b/**/*.tmp.txt` to glob, the path above will be returned twice
        3. `**` will match any matched file, directory, symlink and '' by default, when recursive is `True`
        4. fs_glob returns same as glob.glob(pathname, recursive=True) in acsending alphabetical order.
        5. Hidden files (filename stars with '.') will not be found in the result

        :param pattern: Glob the given relative pattern in the directory represented by this path
        :param recursive: If False, `**` will not search directory recursively
        :param missing_ok: If False and target path doesn't match any file, raise FileNotFoundError
        :returns: An iterator contains paths match `pathname`
        """
        glob_path = self.path_with_protocol
        if pattern:
            glob_path = self.joinpath(pattern).path_with_protocol

        def _scandir(dirname: str) -> Iterator[Tuple[str, bool]]:
            for entry in self.from_path(dirname).scandir():
                yield entry.name, entry.is_dir()

        def _exist(path: PathLike, followlinks: bool = False):
            return self.from_path(path).exists(followlinks=followlinks)

        def _is_dir(path: PathLike, followlinks: bool = False):
            return self.from_path(path).is_dir(followlinks=followlinks)

        fs = FSFunc(_exist, _is_dir, _scandir)
        for real_path in _create_missing_ok_generator(
            iglob(fspath(glob_path), recursive=recursive, fs=fs),
            missing_ok,
            FileNotFoundError("No match any file: %r" % glob_path),
        ):
            yield self.from_path(real_path)

    def is_dir(self, followlinks: bool = False) -> bool:
        """
        Test if a path is directory

        .. note::

            The difference between this function and ``os.path.isdir`` is that this function regard symlink as file

        :param followlinks: False if regard symlink as file, else True
        :returns: True if the path is a directory, else False

        """
        try:
            stat = self.stat(follow_symlinks=followlinks)
            if S_ISDIR(stat.st_mode):
                return True
        except FileNotFoundError:
            pass
        return False

    def is_file(self, followlinks: bool = False) -> bool:
        """
        Test if a path is file

        .. note::

            The difference between this function and ``os.path.isfile`` is that this function regard symlink as file

        :param followlinks: False if regard symlink as file, else True
        :returns: True if the path is a file, else False

        """
        try:
            stat = self.stat(follow_symlinks=followlinks)
            if S_ISREG(stat.st_mode):
                return True
        except FileNotFoundError:
            pass
        return False

    def listdir(self) -> List[str]:
        """
        Get all contents of given sftp path. The result is in acsending alphabetical order.

        :returns: All contents have in the path in acsending alphabetical order
        """
        if not self.is_dir():
            raise NotADirectoryError(f"Not a directory: '{self.path_with_protocol}'")
        return sorted(self._client.listdir(self._real_path))

    def iterdir(self) -> Iterator["SftpPath"]:
        """
        Get all contents of given sftp path. The result is in acsending alphabetical order.

        :returns: All contents have in the path in acsending alphabetical order
        """
        if not self.is_dir():
            raise NotADirectoryError(f"Not a directory: '{self.path_with_protocol}'")
        for path in self.listdir():
            yield self.joinpath(path)  # type: ignore

    def load(self) -> BinaryIO:
        """Read all content on specified path and write into memory

        User should close the BinaryIO manually

        :returns: Binary stream
        """
        with self.open(mode="rb") as f:
            data = f.read()
        return io.BytesIO(data)

    def mkdir(self, mode=0o777, parents: bool = False, exist_ok: bool = False):
        """
        make a directory on sftp, including parent directory

        If there exists a file on the path, raise FileExistsError

        :param mode: If mode is given, it is combined with the process’ umask value to determine the file mode and access flags.
        :param parents: If parents is true, any missing parents of this path are created as needed;
        If parents is false (the default), a missing parent raises FileNotFoundError.
        :param exist_ok: If False and target directory exists, raise FileExistsError
        :raises: FileExistsError
        """
        if self.exists():
            if not exist_ok:
                raise FileExistsError(f"File exists: '{self.path_with_protocol}'")
            return

        if parents:
            parent_path_objects = []
            for parent_path_object in self.parents:
                if parent_path_object.exists():
                    break
                else:
                    parent_path_objects.append(parent_path_object)
            for parent_path_object in parent_path_objects[::-1]:
                parent_path_object.mkdir(mode=mode, parents=False, exist_ok=True)
        try:
            self._client.mkdir(path=self._real_path, mode=mode)
        except OSError:
            # catch OSError when mkdir concurrently
            if not self.exists():
                raise

    def realpath(self) -> str:
        """Return the real path of given path

        :returns: Real path of given path
        """
        return self.resolve().path_with_protocol

    def _is_same_backend(self, other: "SftpPath") -> bool:
        return (
            self._urlsplit_parts.hostname == other._urlsplit_parts.hostname
            and self._urlsplit_parts.username == other._urlsplit_parts.username
            and self._urlsplit_parts.password == other._urlsplit_parts.password
            and self._urlsplit_parts.port == other._urlsplit_parts.port
        )

    def _is_same_protocol(self, path):
        return is_sftp(path)

    def rename(self, dst_path: PathLike) -> "SftpPath":
        """
        rename file on sftp

        :param dst_path: Given destination path
        """
        if not self._is_same_protocol(dst_path):
            raise OSError("Not a %s path: %r" % (self.protocol, dst_path))
        if str(dst_path).endswith("/"):
            raise IsADirectoryError("Is a directory: %r" % dst_path)

        dst_path = self.from_path(dst_path)
        src_stat = self.stat()

        if self._is_same_backend(dst_path):
            try:
                self._client.rename(self._real_path, dst_path._real_path)
            except OSError:
                if dst_path.exists():
                    raise FileExistsError("File exists: %s" % dst_path)
        else:
            if self.is_dir():
                for file_entry in self.scandir():
                    self.from_path(file_entry.path).rename(dst_path.joinpath(file_entry.name))
            else:
                with self.open("rb") as fsrc:
                    with dst_path.open("wb") as fdst:
                        length = 16 * 1024
                        while True:
                            buf = fsrc.read(length)
                            if not buf:
                                break
                            fdst.write(buf)
                self.unlink()

        dst_path.utime(src_stat.st_atime, src_stat.st_mtime)
        dst_path.chmod(src_stat.st_mode)
        return dst_path

    def replace(self, dst_path: PathLike) -> "SftpPath":
        """
        move file on sftp

        :param dst_path: Given destination path
        """
        return self.rename(dst_path=dst_path)

    def remove(self, missing_ok: bool = False) -> None:
        """
        Remove the file or directory on sftp

        :param missing_ok: if False and target file/directory not exists, raise FileNotFoundError
        """
        if missing_ok and not self.exists():
            return
        if self.is_dir():
            for file_entry in self.scandir():
                self.from_path(file_entry.path).remove(missing_ok=missing_ok)
            self._client.rmdir(self._real_path)
        else:
            self._client.unlink(self._real_path)

    def scan(self, missing_ok: bool = True, followlinks: bool = False) -> Iterator[str]:
        """
        Iteratively traverse only files in given directory, in alphabetical order.
        Every iteration on generator yields a path string.

        If path is a file path, yields the file only
        If path is a non-existent path, return an empty generator
        If path is a bucket path, return all file paths in the bucket

        :param missing_ok: If False and there's no file in the directory, raise FileNotFoundError
        :returns: A file path generator
        """
        scan_stat_iter = self.scan_stat(missing_ok=missing_ok, followlinks=followlinks)

        for file_entry in scan_stat_iter:
            yield file_entry.path

    def scan_stat(self, missing_ok: bool = True, followlinks: bool = False) -> Iterator[FileEntry]:
        """
        Iteratively traverse only files in given directory, in alphabetical order.
        Every iteration on generator yields a tuple of path string and file stat

        :param missing_ok: If False and there's no file in the directory, raise FileNotFoundError
        :returns: A file path generator
        """

        def create_generator() -> Iterator[FileEntry]:

            try:
                stat = self.stat(follow_symlinks=followlinks)
            except FileNotFoundError:
                return
            if S_ISREG(stat.st_mode):
                yield FileEntry(self.name, self.path_with_protocol, self.stat(follow_symlinks=followlinks))
                return

            for name in self.listdir():
                current_path = self.joinpath(name)
                if current_path.is_dir():
                    yield from current_path.scan_stat(
                        missing_ok=missing_ok,
                        followlinks=followlinks,
                    )
                else:
                    yield FileEntry(
                        current_path.name,  # type: ignore
                        current_path.path_with_protocol,
                        current_path.stat(follow_symlinks=followlinks),
                    )

        return _create_missing_ok_generator(
            create_generator(), missing_ok, FileNotFoundError("No match any file in: %r" % self.path_with_protocol)
        )

    def scandir(self) -> Iterator[FileEntry]:
        """
        Get all content of given file path.

        :returns: An iterator contains all contents have prefix path
        """
        if not self.exists():
            raise FileNotFoundError("No such directory: %r" % self.path_with_protocol)

        if not self.is_dir():
            raise NotADirectoryError("Not a directory: %r" % self.path_with_protocol)

        def create_generator():
            for name in self.listdir():
                current_path = self.joinpath(name)
                yield FileEntry(
                    current_path.name, current_path.path_with_protocol, current_path.lstat()  # type: ignore
                )  # type: ignore

        return ContextIterator(create_generator())

    def stat(self, follow_symlinks=True) -> StatResult:
        """
        Get StatResult of file on sftp, including file size and mtime, referring to fs_getsize and fs_getmtime

        :returns: StatResult
        """
        if follow_symlinks:
            result = _make_stat(self._client.stat(self._real_path))
        else:
            result = _make_stat(self._client.lstat(self._real_path))
        return result

    def unlink(self, missing_ok: bool = False) -> None:
        """
        Remove the file on sftp

        :param missing_ok: if False and target file not exists, raise FileNotFoundError
        """
        if missing_ok and not self.exists():
            return
        self._client.unlink(self._real_path)

    def walk(self, followlinks: bool = False) -> Iterator[Tuple[str, List[str], List[str]]]:
        """
        Generate the file names in a directory tree by walking the tree top-down.
        For each directory in the tree rooted at directory path (including path itself),
        it yields a 3-tuple (root, dirs, files).

        root: a string of current path
        dirs: name list of subdirectories (excluding '.' and '..' if they exist) in 'root'. The list is sorted by ascending alphabetical order
        files: name list of non-directory files (link is regarded as file) in 'root'. The list is sorted by ascending alphabetical order

        If path not exists, or path is a file (link is regarded as file), return an empty generator

        .. note::

            Be aware that setting ``followlinks`` to True can lead to infinite recursion if a link points to a parent directory of itself. fs_walk() does not keep track of the directories it visited already.

        :param followlinks: False if regard symlink as file, else True
        :returns: A 3-tuple generator
        """
        if not self.exists(followlinks=followlinks):
            return

        if self.is_file(followlinks=followlinks):
            return

        stack = [self._real_path]
        while stack:
            root = stack.pop()
            dirs, files = [], []
            filenames = self._client.listdir(root)
            for name in filenames:
                current_path = self._generate_path_object(root).joinpath(name)
                if current_path.is_file(followlinks=followlinks):
                    files.append(name)
                elif current_path.is_dir(followlinks=followlinks):
                    dirs.append(name)

            dirs = sorted(dirs)
            files = sorted(files)

            yield self._generate_path_object(root).path_with_protocol, dirs, files

            stack.extend((os.path.join(root, directory) for directory in reversed(dirs)))

    def resolve(self, strict=False) -> "SftpPath":
        """Equal to sftp_realpath

        :param strict: Ignore this parameter, just for compatibility
        :return: Return the canonical path of the specified filename, eliminating any symbolic links encountered in the path.
        :rtype: SftpPath
        """
        path = self._client.normalize(self._real_path)
        return self._generate_path_object(path, resolve=True)

    def md5(self, recalculate: bool = False, followlinks: bool = True):
        """
        Calculate the md5 value of the file

        :param recalculate: Ignore this parameter, just for compatibility
        :param followlinks: Ignore this parameter, just for compatibility
        returns: md5 of file
        """
        if self.is_dir():
            hash_md5 = hashlib.md5()  # nosec
            for file_name in self.listdir():
                chunk = self.joinpath(file_name).md5(recalculate=recalculate, followlinks=followlinks).encode()  # type: ignore
                hash_md5.update(chunk)
            return hash_md5.hexdigest()
        with self.open("rb") as src:  # type: ignore
            md5 = calculate_md5(src)
        return md5

    def symlink(self, dst_path: PathLike) -> None:
        """
        Create a symbolic link pointing to src_path named dst_path.

        :param dst_path: Desination path
        """
        dst_path = self.from_path(dst_path)
        if dst_path.exists(followlinks=False):
            raise FileExistsError(f"File exists: '{dst_path.path_with_protocol}'")
        return self._client.symlink(self._real_path, dst_path._real_path)

    def readlink(self) -> "SftpPath":
        """
        Return a SftpPath instance representing the path to which the symbolic link points.
        :returns: Return a SftpPath instance representing the path to which the symbolic link points.
        """
        if not self.is_symlink():
            raise OSError("Not a symlink: %s" % self.path_with_protocol)
        path = self._client.readlink(self._real_path)
        if not path.startswith("/"):
            return self.parent.joinpath(path)
        return self._generate_path_object(path)

    def is_symlink(self) -> bool:
        """Test whether a path is a symbolic link

        :return: If path is a symbolic link return True, else False
        :rtype: bool
        """
        return self.lstat().is_symlink()

    def cwd(self) -> "SftpPath":
        """Return current working directory

        returns: Current working directory
        """
        return self._generate_path_object(self._client.normalize("."))

    def save(self, file_object: BinaryIO):
        """Write the opened binary stream to path
        If parent directory of path doesn't exist, it will be created.

        :param file_object: stream to be read
        """
        with self.open(mode="wb") as output:
            output.write(file_object.read())

    def open(
        self, mode: str = "r", buffering=-1, encoding: Optional[str] = None, errors: Optional[str] = None, **kwargs
    ) -> IO[AnyStr]:  # pytype: disable=signature-mismatch
        """Open a file on the path.

        :param mode: Mode to open file
        :param buffering: buffering is an optional integer used to set the buffering policy.
        :param encoding: encoding is the name of the encoding used to decode or encode the file. This should only be used in text mode.
        :param errors: errors is an optional string that specifies how encoding and decoding errors are to be handled—this cannot be used in binary mode.
        :returns: File-Like object
        """
        if "w" in mode or "x" in mode or "a" in mode:
            if self.is_dir():
                raise IsADirectoryError("Is a directory: %r" % self.path_with_protocol)
            self.parent.mkdir(parents=True, exist_ok=True)
        elif not self.exists():
            raise FileNotFoundError("No such file: %r" % self.path_with_protocol)
        fileobj = self._client.open(self._real_path, mode, bufsize=buffering)
        fileobj.name = self.path
        if "r" in mode and "b" not in mode:
            return io.TextIOWrapper(fileobj, encoding=encoding, errors=errors)  # type: ignore
        return fileobj  # type: ignore

    def chmod(self, mode: int, follow_symlinks: bool = True):
        """
        Change the file mode and permissions, like os.chmod().

        :param mode: the file mode you want to change
        :param followlinks: Ignore this parameter, just for compatibility
        """
        return self._client.chmod(path=self._real_path, mode=mode)

    def absolute(self) -> "SftpPath":
        """
        Make the path absolute, without normalization or resolving symlinks. Returns a new path object
        """
        return self.resolve()

    def rmdir(self):
        """
        Remove this directory. The directory must be empty.
        """
        if len(self.listdir()) > 0:
            raise OSError(f"Directory not empty: '{self.path_with_protocol}'")
        return self._client.rmdir(self._real_path)

    def _exec_command(
        self,
        command: List[str],
        bufsize: int = -1,
        timeout: Optional[int] = None,
        environment: Optional[dict] = None,
    ) -> subprocess.CompletedProcess:
        with get_ssh_session(
            hostname=self._urlsplit_parts.hostname,
            port=self._urlsplit_parts.port,
            username=self._urlsplit_parts.username,
            password=self._urlsplit_parts.password,
        ) as chan:
            chan.settimeout(timeout)
            if environment:
                chan.update_environment(environment)
            chan.exec_command(" ".join([shlex.quote(arg) for arg in command]))
            stdout = chan.makefile("r", bufsize).read().decode(errors="backslashreplace")
            stderr = chan.makefile_stderr("r", bufsize).read().decode(errors="backslashreplace")
            returncode = chan.recv_exit_status()
        return subprocess.CompletedProcess(args=command, returncode=returncode, stdout=stdout, stderr=stderr)

    def copy(self, dst_path: PathLike, callback: Optional[Callable[[int], None]] = None, followlinks: bool = False):
        """
        Copy the file to the given destination path.

        :param dst_path: The destination path to copy the file to.
        :param callback: An optional callback function that takes an integer parameter and is called
                        periodically during the copy operation to report the number of bytes copied.
        :param followlinks: Whether to follow symbolic links when copying directories.
        :raises IsADirectoryError: If the source is a directory.
        :raises OSError: If there is an error copying the file.
        """
        if followlinks and self.is_symlink():
            return self.readlink().copy(dst_path=dst_path, callback=callback)

        if not self._is_same_protocol(dst_path):
            raise OSError("Not a %s path: %r" % (self.protocol, dst_path))
        if str(dst_path).endswith("/"):
            raise IsADirectoryError("Is a directory: %r" % dst_path)

        if self.is_dir():
            raise IsADirectoryError("Is a directory: %r" % self.path_with_protocol)

        self.from_path(os.path.dirname(dst_path)).makedirs(exist_ok=True)
        dst_path = self.from_path(dst_path)
        if self._is_same_backend(dst_path):
            if self._real_path == dst_path._real_path:
                raise SameFileError(f"'{self.path}' and '{dst_path.path}' are the same file")
            exec_result = self._exec_command(["cp", self._real_path, dst_path._real_path])
            if exec_result.returncode != 0:
                _logger.error(exec_result.stderr)
                raise OSError(f"Copy file error, returncode: {exec_result.returncode}")
            if callback:
                callback(self.stat(follow_symlinks=followlinks).size)
        else:
            with self.open("rb") as fsrc:
                with dst_path.open("wb") as fdst:
                    length = 16 * 1024
                    while True:
                        buf = fsrc.read(length)
                        if not buf:
                            break
                        fdst.write(buf)
                        if callback:
                            callback(len(buf))

        src_stat = self.stat()
        dst_path.utime(src_stat.st_atime, src_stat.st_mtime)
        dst_path._client.chmod(dst_path._real_path, src_stat.st_mode)

    def sync(self, dst_path: PathLike, followlinks: bool = False, force: bool = False):
        """Copy file/directory on src_url to dst_url

        :param dst_url: Given destination path
        :param followlinks: False if regard symlink as file, else True
        :param force: Sync file forcely, do not ignore same files
        """
        if not self._is_same_protocol(dst_path):
            raise OSError("Not a %s path: %r" % (self.protocol, dst_path))

        for src_file_path, dst_file_path in _sftp_scan_pairs(self.path_with_protocol, dst_path):
            dst_path = self.from_path(dst_file_path)
            src_path = self.from_path(src_file_path)
            if not force and dst_path.exists() and is_same_file(src_path.stat(), dst_path.stat(), "copy"):
                continue
            self.from_path(os.path.dirname(dst_file_path)).mkdir(parents=True, exist_ok=True)
            self.from_path(src_file_path).copy(dst_file_path, followlinks=followlinks)

    def utime(self, atime: Union[float, int], mtime: Union[float, int]) -> None:
        """
        Set the access and modified times of the file specified by path.

        :param atime: The access time to be set.
        :type atime: Union[float, int]
        :param mtime: The modification time to be set.
        :type mtime: Union[float, int]
        :return: None
        """
        return self._client.utime(self._real_path, (atime, mtime))
