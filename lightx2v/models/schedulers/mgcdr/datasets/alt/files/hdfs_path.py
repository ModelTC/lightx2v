# Standard Library
import hashlib
import io
import os
import sys
from functools import lru_cache
from typing import IO, AnyStr, BinaryIO, Iterator, List, Optional, Tuple

# Import from alt
from alt.files.errors import _create_missing_ok_generator, raise_hdfs_error
from alt.files.interfaces import FileEntry, PathLike, StatResult, URIPath
from alt.files.lib.compat import fspath
from alt.files.lib.glob import FSFunc, iglob
from alt.files.lib.hdfs_prefetch_reader import HdfsPrefetchReader
from alt.files.lib.hdfs_tools import hdfs_api
from alt.files.lib.url import get_url_scheme
from alt.files.smart_path import SmartPath
from alt.files.utils import cachedproperty

__all__ = [
    "HdfsPath",
    "is_hdfs",
    "hdfs_glob",
    "hdfs_glob_stat",
    "hdfs_iglob",
    "hdfs_makedirs",
]

HDFS_USER = "HDFS_USER"
HDFS_URL = "HDFS_URL"
HDFS_ROOT = "HDFS_ROOT"
HDFS_TIMEOUT = "HDFS_TIMEOUT"
HDFS_TOKEN = "HDFS_TOKEN"
HDFS_CONFIG_PATH = "HDFS_CONFIG_PATH"
MAX_RETRIES = 10
DEFAULT_HDFS_TIMEOUT = 10


def is_hdfs(path: PathLike) -> bool:  # pytype: disable=invalid-annotation
    """Test if a path is sftp path

    :param path: Path to be tested
    :returns: True of a path is sftp path, else False
    """
    return fspath(path).startswith("hdfs://")


def get_hdfs_config(profile_name: Optional[str] = None):
    env_profile = f"{profile_name.upper()}__" if profile_name else ""
    config = {
        "user": os.getenv(f"{env_profile}{HDFS_USER}"),
        "url": os.getenv(f"{env_profile}{HDFS_URL}"),
        "root": os.getenv(f"{env_profile}{HDFS_ROOT}"),
        "timeout": DEFAULT_HDFS_TIMEOUT,
        "token": os.getenv(f"{env_profile}{HDFS_TOKEN}"),
    }
    timeout_env = f"{env_profile}{HDFS_TIMEOUT}"
    if os.getenv(timeout_env):
        config["timeout"] = int(os.getenv(timeout_env))

    config_path = os.getenv(HDFS_CONFIG_PATH) or os.path.expanduser("~/.hdfscli.cfg")
    if os.path.exists(config_path):
        all_config = hdfs_api.config.Config(path=config_path)
        if not profile_name:
            if all_config.has_section(all_config.global_section) and all_config.has_option(
                all_config.global_section, "default.alias"
            ):
                profile_name = all_config.get(all_config.global_section, "default.alias")
        for suffix in (".alias", "_alias"):
            section = "{}{}".format(profile_name, suffix)
            if all_config.has_section(section):
                options = dict(all_config.items(section))
                for key, value in config.items():
                    if not value and options.get(key):
                        config[key] = options[key]
                break

    if config["url"]:
        return config

    raise hdfs_api.HdfsError('Config error, please set environments or use "alt.files config hdfs ..."')


@lru_cache()
def get_hdfs_client(profile_name: Optional[str] = None):
    if not hdfs_api:  # pragma: no cover
        raise ImportError("hdfs not found, please `pip install 'alt.files[hdfs]'`")

    config = get_hdfs_config(profile_name)
    if config["token"]:
        config.pop("user", None)
        return hdfs_api.TokenClient(**config)
    config.pop("token", None)
    return hdfs_api.InsecureClient(**config)


def hdfs_glob(
    path: PathLike,
    recursive: bool = True,
    missing_ok: bool = True,
) -> List[str]:
    """Return hdfs path list in ascending alphabetical order, in which path matches glob pattern
    Notes: Only glob in bucket. If trying to match bucket with wildcard characters, raise UnsupportedError

    :param recursive: If False, `**` will not search directory recursively
    :param missing_ok: If False and target path doesn't match any file, raise FileNotFoundError
    :raises: UnsupportedError, when bucket part contains wildcard characters
    :returns: A list contains paths match `path`
    """
    return list(hdfs_iglob(path, recursive=recursive, missing_ok=missing_ok))


def hdfs_glob_stat(path: PathLike, recursive: bool = True, missing_ok: bool = True) -> Iterator[FileEntry]:
    """Return a generator contains tuples of path and file stat, in ascending alphabetical order, in which path matches glob pattern
    Notes: Only glob in bucket. If trying to match bucket with wildcard characters, raise UnsupportedError

    :param recursive: If False, `**` will not search directory recursively
    :param missing_ok: If False and target path doesn't match any file, raise FileNotFoundError
    :raises: UnsupportedError, when bucket part contains wildcard characters
    :returns: A generator contains tuples of path and file stat, in which paths match `path`
    """
    return HdfsPath(path).glob_stat(pattern="", recursive=recursive, missing_ok=missing_ok)


def hdfs_iglob(
    path: PathLike,
    recursive: bool = True,
    missing_ok: bool = True,
) -> Iterator[str]:
    """Return hdfs path iterator in ascending alphabetical order, in which path matches glob pattern
    Notes: Only glob in bucket. If trying to match bucket with wildcard characters, raise UnsupportedError

    :param recursive: If False, `**` will not search directory recursively
    :param missing_ok: If False and target path doesn't match any file, raise FileNotFoundError
    :raises: UnsupportedError, when bucket part contains wildcard characters
    :returns: An iterator contains paths match `path`
    """
    for path_obj in HdfsPath(path).iglob(pattern="", recursive=recursive, missing_ok=missing_ok):
        yield path_obj.path_with_protocol


def hdfs_makedirs(path: PathLike, exist_ok: bool = False):
    """
    Create an hdfs directory.
    Purely creating directory is invalid because it's unavailable on OSS.
    This function is to test the target bucket have WRITE access.

    :param path: Given path
    :param exist_ok: If False and target directory exists, raise S3FileExistsError
    :raises: FileExistsError
    """
    return HdfsPath(path).mkdir(parents=True, exist_ok=exist_ok)


@SmartPath.register
class HdfsPath(URIPath):
    protocol = "hdfs"

    def __init__(self, path: PathLike, *other_paths: PathLike):
        super().__init__(path, *other_paths)
        protocol = get_url_scheme(self.path)
        self._protocol_with_profile = self.protocol
        self._profile_name = None
        if protocol.startswith("hdfs+"):
            self._protocol_with_profile = protocol
            self._profile_name = protocol[5:]

    @property
    def _client(self):
        return get_hdfs_client(profile_name=self._profile_name)

    @cachedproperty
    def path_with_protocol(self) -> str:
        """Return path with protocol, like hdfs://path"""
        path = self.path
        protocol_prefix = self._protocol_with_profile + "://"
        if path.startswith(protocol_prefix):
            return path
        return protocol_prefix + path.lstrip("/")

    @cachedproperty
    def path_without_protocol(self) -> str:
        """Return path without protocol, example: if path is hdfs://path, return path"""
        path = self.path
        protocol_prefix = self._protocol_with_profile + "://"
        if path.startswith(protocol_prefix):
            path = path[len(protocol_prefix) :]
        return path

    @cachedproperty
    def parts(self) -> Tuple[str]:
        """A tuple giving access to the path’s various components"""
        parts = [f"{self._protocol_with_profile}://"]
        path = self.path_without_protocol
        path = path.lstrip("/")
        if path != "":
            parts.extend(path.split("/"))
        return tuple(parts)

    def exists(self, followlinks: bool = False) -> bool:
        """
        Test if path exists

        If the bucket of path are not permitted to read, return False

        :returns: True if path eixsts, else False
        """
        return bool(self._client.status(self.path_without_protocol, strict=False))

    def stat(self, follow_symlinks=True) -> StatResult:
        """
        Get StatResult of path file, including file size and mtime, referring to hdfs_getsize and hdfs_getmtime

        If path is not an existent path, which means hdfs_exist(path) returns False, then raise FileNotFoundError
        If attempt to get StatResult of complete hdfs, such as hdfs_dir_url == 'hdfs://', raise BucketNotFoundError

        :returns: StatResult
        :raises: FileNotFoundError
        """
        with raise_hdfs_error(self.path_with_protocol):
            stat_data = self._client.status(self.path_without_protocol)
            return StatResult(
                size=stat_data["length"],
                mtime=stat_data["modificationTime"] / 1000,
                isdir=stat_data["type"] == "DIRECTORY",
                islnk=False,
                extra=stat_data,
            )

    def getmtime(self, follow_symlinks: bool = False) -> float:
        """
        Get last-modified time of the file on the given path path (in Unix timestamp format).
        If the path is an existent directory, return the latest modified time of all file in it. The mtime of empty directory is 1970-01-01 00:00:00

        If path is not an existent path, which means hdfs_exist(path) returns False, then raise FileNotFoundError

        :returns: Last-modified time
        :raises: FileNotFoundError
        """
        return self.stat(follow_symlinks=follow_symlinks).mtime

    def getsize(self, follow_symlinks: bool = False) -> int:
        """
        Get file size on the given path path (in bytes).
        If the path in a directory, return the sum of all file size in it, including file in subdirectories (if exist).
        The result excludes the size of directory itself. In other words, return 0 Byte on an empty directory path.

        If path is not an existent path, which means hdfs_exist(path) returns False, then raise FileNotFoundError

        :returns: File size
        :raises: FileNotFoundError
        """
        return self.stat(follow_symlinks=follow_symlinks).size

    def glob(
        self,
        pattern,
        recursive: bool = True,
        missing_ok: bool = True,
    ) -> List["HdfsPath"]:
        """Return hdfs path list, in which path matches glob pattern
        Notes: Only glob in bucket. If trying to match bucket with wildcard characters, raise UnsupportedError

        :param pattern: Glob the given relative pattern in the directory represented by this path
        :param recursive: If False, `**` will not search directory recursively
        :param missing_ok: If False and target path doesn't match any file, raise FileNotFoundError
        :raises: UnsupportedError, when bucket part contains wildcard characters
        :returns: A list contains paths match `hdfs_pathname`
        """
        return list(self.iglob(pattern=pattern, recursive=recursive, missing_ok=missing_ok))

    def glob_stat(self, pattern, recursive: bool = True, missing_ok: bool = True) -> Iterator[FileEntry]:
        """Return a generator contains tuples of path and file stat, in which path matches glob pattern
        Notes: Only glob in bucket. If trying to match bucket with wildcard characters, raise UnsupportedError

        :param pattern: Glob the given relative pattern in the directory represented by this path
        :param recursive: If False, `**` will not search directory recursively
        :param missing_ok: If False and target path doesn't match any file, raise FileNotFoundError
        :raises: UnsupportedError, when bucket part contains wildcard characters
        :returns: A generator contains tuples of path and file stat, in which paths match `hdfs_pathname`
        """
        for path_obj in self.iglob(pattern=pattern, recursive=recursive, missing_ok=missing_ok):
            yield FileEntry(path_obj.name, path_obj.path, path_obj.stat())

    def iglob(
        self,
        pattern,
        recursive: bool = True,
        missing_ok: bool = True,
    ) -> Iterator["HdfsPath"]:
        """Return hdfs path iterator, in which path matches glob pattern
        Notes: Only glob in bucket. If trying to match bucket with wildcard characters, raise UnsupportedError

        :param pattern: Glob the given relative pattern in the directory represented by this path
        :param recursive: If False, `**` will not search directory recursively
        :param missing_ok: If False and target path doesn't match any file, raise FileNotFoundError
        :raises: UnsupportedError, when bucket part contains wildcard characters
        :returns: An iterator contains paths match `hdfs_pathname`
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

        fs_func = FSFunc(_exist, _is_dir, _scandir)
        for real_path in _create_missing_ok_generator(
            iglob(fspath(glob_path), recursive=recursive, fs=fs_func),
            missing_ok,
            FileNotFoundError("No match any file: %r" % glob_path),
        ):
            yield self.from_path(real_path)

    def is_dir(self, followlinks: bool = False) -> bool:
        """
        Test if an hdfs url is directory
        Specific procedures are as follows:
        If there exists a suffix, of which ``os.path.join(path, suffix)`` is a file
        If the url is empty bucket or hdfs://

        :param followlinks: whether followlinks is True or False, result is the same. Because hdfs symlink not support dir.
        :returns: True if path is hdfs directory, else False
        """
        return self.stat().is_dir()

    def is_file(self, followlinks: bool = False) -> bool:
        """
        Test if an path is file

        :returns: True if path is hdfs file, else False
        """
        return self.stat().is_file()

    def listdir(self, followlinks: bool = False) -> List[str]:
        """
        Get all contents of given path.

        :returns: All contents have prefix of path.
        :raises: FileNotFoundError, NotADirectoryError
        """
        if not self.is_dir():
            raise NotADirectoryError("Not a directory: %r" % self.path)
        with raise_hdfs_error(self.path_with_protocol):
            return self._client.list(self.path_without_protocol)

    def iterdir(self, followlinks: bool = False) -> Iterator["HdfsPath"]:
        """
        Get all contents of given path.

        :returns: All contents have prefix of path.
        :raises: FileNotFoundError, NotADirectoryError
        """
        for filename in self.listdir(followlinks=followlinks):
            yield self.joinpath(filename)  # pytype: disable=bad-return-type

    def load(self, followlinks: bool = False) -> BinaryIO:
        """Read all content in binary on specified path and write into memory

        User should close the BinaryIO manually

        :returns: BinaryIO
        """

        buffer = io.BytesIO()
        with self.open("rb") as f:
            buffer.write(f.read())
        buffer.seek(0)
        return buffer

    def mkdir(self, mode=0o777, parents: bool = False, exist_ok: bool = False):
        """
        Create an hdfs directory.
        Purely creating directory is invalid because it's unavailable on OSS.
        This function is to test the target bucket have WRITE access.

        :param mode: Octal permission to set on the newly created directory.
            These permissions will only be set on directories that do not already exist.
        :param parents: parents is ignored, only be compatible with pathlib.Path
        :param exist_ok: If False and target directory exists, raise FileExistsError
        :raises: BucketNotFoundError, FileExistsError
        """
        if not exist_ok and self.exists():
            raise FileExistsError("File exists: %r" % self.path)
        with raise_hdfs_error(self.path_with_protocol):
            self._client.makedirs(self.path_without_protocol, permission=mode)

    def rename(self, dst_path: PathLike) -> "HdfsPath":
        """
        Move hdfs file path from src_path to dst_path

        :param dst_path: Given destination path
        """
        dst_path = self.from_path(dst_path)
        with raise_hdfs_error(self.path_with_protocol):
            self._client.rename(self.path_without_protocol, dst_path.path_without_protocol)
        return dst_path

    def move(self, dst_path: PathLike) -> None:
        """
        Move file/directory path from src_path to dst_path

        :param dst_path: Given destination path
        """
        self.rename(dst_path=dst_path)

    def remove(self, missing_ok: bool = False) -> None:
        """
        Remove the file or directory on hdfs, `hdfs://` and `hdfs://bucket` are not permitted to remove

        :param missing_ok: if False and target file/directory not exists, raise FileNotFoundError
        :raises: FileNotFoundError, UnsupportedError
        """
        try:
            with raise_hdfs_error(self.path_with_protocol):
                self._client.delete(self.path_without_protocol, recursive=True)
        except Exception as e:
            if not missing_ok or not isinstance(e, FileNotFoundError):
                raise

    def scan(self, missing_ok: bool = True, followlinks: bool = False) -> Iterator[str]:
        """
        Iteratively traverse only files in given hdfs directory.
        Every iteration on generator yields a path string.

        If path is a file path, yields the file only
        If path is a non-existent path, return an empty generator
        If path is a bucket path, return all file paths in the bucket
        If path is an empty bucket, return an empty generator
        If path doesn't contain any bucket, which is path == 'hdfs://', raise UnsupportedError. walk() on complete hdfs is not supported in alt.files

        :param missing_ok: If False and there's no file in the directory, raise FileNotFoundError
        :raises: UnsupportedError
        :returns: A file path generator
        """
        for file_entry in self.scan_stat(missing_ok=missing_ok, followlinks=followlinks):
            yield file_entry.path

    def scan_stat(self, missing_ok: bool = True, followlinks: bool = False) -> Iterator[FileEntry]:
        """
        Iteratively traverse only files in given directory.
        Every iteration on generator yields a tuple of path string and file stat

        :param missing_ok: If False and there's no file in the directory, raise FileNotFoundError
        :raises: UnsupportedError
        :returns: A file path generator
        """
        with raise_hdfs_error(self.path_with_protocol):
            for (root, _root_status), _dir_infos, file_infos in self._client.walk(
                self.path_without_protocol, status=True, ignore_missing=missing_ok
            ):
                for filename, stat_data in file_infos:
                    yield FileEntry(
                        name=filename,
                        path=self.from_path(f"{self._protocol_with_profile}://{root.lstrip('/')}")
                        .joinpath(filename)
                        .path_with_protocol,
                        stat=StatResult(
                            size=stat_data["length"],
                            mtime=stat_data["modificationTime"] / 1000,
                            isdir=False,
                            islnk=False,
                            extra=stat_data,
                        ),
                    )

    def scandir(self, followlinks: bool = False) -> Iterator[FileEntry]:
        """
        Get all contents of given path, the order of result is not guaranteed.

        :returns: All contents have prefix of path
        :raises: FileNotFoundError, NotADirectoryError
        """
        with raise_hdfs_error(self.path_with_protocol):
            for filename, stat_data in self._client.list(self.path_without_protocol, status=True):
                yield FileEntry(
                    name=filename,
                    path=self.joinpath(filename).path_with_protocol,
                    stat=StatResult(
                        size=stat_data["length"],
                        mtime=stat_data["modificationTime"] / 1000,
                        isdir=stat_data["type"] == "DIRECTORY",
                        islnk=False,
                        extra=stat_data,
                    ),
                )

    def unlink(self, missing_ok: bool = False) -> None:
        """
        Remove the file on hdfs

        :param missing_ok: if False and target file not exists, raise FileNotFoundError
        :raises: FileNotFoundError, IsADirectoryError
        """
        if self.is_dir():
            raise IsADirectoryError("Path is a directory: %r" % self.path)
        self.remove(missing_ok=missing_ok)

    def walk(self, followlinks: bool = False) -> Iterator[Tuple[str, List[str], List[str]]]:
        """
        Iteratively traverse the given hdfs directory, in top-bottom order. In other words, firstly traverse parent directory, if subdirectories exist, traverse the subdirectories.
        Every iteration on generator yields a 3-tuple: (root, dirs, files)

        - root: Current hdfs path;
        - dirs: Name list of subdirectories in current directory.
        - files: Name list of files in current directory.

        If path is a file path, return an empty generator
        If path is a non-existent path, return an empty generator
        If path is a bucket path, bucket will be the top directory, and will be returned at first iteration of generator
        If path is an empty bucket, only yield one 3-tuple (notes: hdfs doesn't have empty directory)
        If path doesn't contain any bucket, which is path == 'hdfs://', raise UnsupportedError. walk() on complete hdfs is not supported in alt.files

        :param followlinks: whether followlinks is True or False, result is the same. Because hdfs not support symlink.
        :returns: A 3-tuple generator
        """
        with raise_hdfs_error(self.path_with_protocol):
            for path, dirs, files in self._client.walk(self.path_without_protocol, ignore_missing=True, allow_dir_changes=True):
                yield f"{self._protocol_with_profile}://{path.lstrip('/')}", dirs, files

    def md5(self, recalculate: bool = False, followlinks: bool = False) -> str:
        """
        Get checksum of the file or dir.

        :param recalculate: Ignore this parameter, just for compatibility
        :param followlinks: Ignore this parameter, just for compatibility
        :returns: checksum
        """
        if self.is_dir(followlinks=followlinks):
            hash_md5 = hashlib.md5()  # nosec
            for file_name in self.listdir():
                chunk = self.joinpath(file_name).md5(recalculate=recalculate).encode()  # pytype: disable=attribute-error
                hash_md5.update(chunk)
            return hash_md5.hexdigest()
        with raise_hdfs_error(self.path_with_protocol):
            return self._client.checksum(self.path_without_protocol)["bytes"]

    def save(self, file_object: BinaryIO):
        """Write the opened binary stream to specified path, but the stream won't be closed

        :param file_object: Stream to be read
        """
        with raise_hdfs_error(self.path_with_protocol):
            self._client.write(self.path_without_protocol, overwrite=True, data=file_object)

    def open(
        self,
        mode: str = "r",
        *,
        buffering: Optional[int] = None,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        **kwargs,
    ) -> IO[AnyStr]:  # pytype: disable=signature-mismatch
        if "+" in mode:
            raise ValueError("unacceptable mode: %r" % mode)

        if "b" in mode:
            encoding = None
        elif not encoding:
            encoding = sys.getdefaultencoding()

        with raise_hdfs_error(self.path_with_protocol):
            if mode in ("r", "rb"):
                keys = ["block_size", "block_capacity", "block_forward", "max_retries", "max_workers"]
                input_kwargs = {}
                for key in keys:
                    if key in kwargs:
                        input_kwargs[key] = kwargs[key]
                file_obj = HdfsPrefetchReader(
                    hdfs_path=self.path_without_protocol, client=self._client, profile_name=self._profile_name, **input_kwargs
                )
                if "b" not in mode:
                    file_obj = io.TextIOWrapper(file_obj, encoding=encoding, errors=errors)  # pytype: disable=wrong-arg-types
                    file_obj.mode = mode
                return file_obj  # pytype: disable=bad-return-type
            elif mode in ("w", "wb"):
                return self._client.write(  # pytype: disable=bad-return-type
                    self.path_without_protocol, overwrite=True, buffersize=buffering, encoding=encoding
                )
            elif mode in ("a", "ab"):
                return self._client.write(  # pytype: disable=bad-return-type
                    self.path_without_protocol, append=True, buffersize=buffering, encoding=encoding
                )
        raise ValueError("unacceptable mode: %r" % mode)

    def absolute(self) -> "HdfsPath":
        """
        Make the path absolute, without normalization or resolving symlinks. Returns a new path object
        """
        with raise_hdfs_error(self.path_with_protocol):
            real_path = self._client.resolve(self.path_without_protocol)
            return self.from_path(f"{self._protocol_with_profile}:///{real_path.lstrip('/')}")
