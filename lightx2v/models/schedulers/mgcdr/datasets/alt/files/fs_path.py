# Standard Library
import hashlib
import io
import os
import pathlib
import shutil
from shutil import copytree
from stat import S_ISDIR as stat_isdir
from stat import S_ISLNK as stat_islnk
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

# Import from alt
from alt.files.errors import _create_missing_ok_generator
from alt.files.interfaces import (
    Access,
    ContextIterator,
    FileEntry,
    PathLike,
    StatResult,
    URIPath
)
from alt.files.lib.compare import is_same_file
from alt.files.lib.glob import iglob
from alt.files.lib.url import get_url_scheme
from alt.files.utils import cachedproperty, calculate_md5

# Import from local
from .lib.compat import fspath
from .lib.joinpath import path_join
from .smart_path import SmartPath

__all__ = [
    "FSPath",
    "is_fs",
    "StatResult",
    "fs_path_join",
    "_make_stat",
    "fs_readlink",
    "fs_cwd",
    "fs_home",
    "fs_iglob",
    "fs_glob",
    "fs_glob_stat",
    "fs_rename",
    "fs_resolve",
    "fs_move",
    "fs_makedirs",
    "fs_lstat",
]


def _make_stat(stat: os.stat_result) -> StatResult:
    return StatResult(
        size=stat.st_size,
        ctime=stat.st_ctime,
        mtime=stat.st_mtime,
        isdir=stat_isdir(stat.st_mode),
        islnk=stat_islnk(stat.st_mode),
        extra=stat,
    )


def is_fs(path: Union["PathLike", int]) -> bool:
    """Test if a path is fs path

    :param path: Path to be tested
    :returns: True of a path is fs path, else False
    """
    if isinstance(path, int):
        return True
    path = fspath(path)
    scheme = get_url_scheme(path)
    return scheme == "" or scheme == "file"


def fs_path_join(path: PathLike, *other_paths: PathLike) -> str:
    return path_join(fspath(path), *map(fspath, other_paths))


def fs_readlink(path) -> str:
    """
    Return a string representing the path to which the symbolic link points.
    :returns: Return a string representing the path to which the symbolic link points.
    """
    return os.readlink(path)


def fs_cwd() -> str:
    """Return current working directory

    returns: Current working directory
    """
    return os.getcwd()


def fs_home():
    """Return the home directory

    returns: Home directory path
    """
    return os.path.expanduser("~")


def fs_iglob(path: PathLike, recursive: bool = True, missing_ok: bool = True) -> Iterator[str]:
    """Return path iterator in ascending alphabetical order, in which path matches glob pattern

    1. If doesn't match any path, return empty list
        Notice:  ``glob.glob`` in standard library returns ['a/'] instead of empty list when pathname is like `a/**`, recursive is True and directory 'a' doesn't exist. fs_glob behaves like ``glob.glob`` in standard library under such circumstance.
    2. No guarantee that each path in result is different, which means:
        Assume there exists a path `/a/b/c/b/d.tmp.txt`
        use path pattern like `/**/b/**/*.tmp.txt` to glob, the path above will be returned twice
    3. `**` will match any matched file, directory, symlink and '' by default, when recursive is `True`
    4. fs_glob returns same as glob.glob(pathname, recursive=True) in acsending alphabetical order.
    5. Hidden files (filename stars with '.') will not be found in the result

    :param recursive: If False, `**` will not search directory recursively
    :param missing_ok: If False and target path doesn't match any file, raise FileNotFoundError
    :returns: An iterator contains paths match `pathname`
    """
    for path in _create_missing_ok_generator(
        iglob(fspath(path), recursive=recursive), missing_ok, FileNotFoundError("No match any file: %r" % path)
    ):
        yield path


def fs_glob(path: PathLike, recursive: bool = True, missing_ok: bool = True) -> List[str]:
    """Return path list in ascending alphabetical order, in which path matches glob pattern

    1. If doesn't match any path, return empty list
        Notice:  ``glob.glob`` in standard library returns ['a/'] instead of empty list when pathname is like `a/**`, recursive is True and directory 'a' doesn't exist. fs_glob behaves like ``glob.glob`` in standard library under such circumstance.
    2. No guarantee that each path in result is different, which means:
        Assume there exists a path `/a/b/c/b/d.tmp.txt`
        use path pattern like `/**/b/**/*.tmp.txt` to glob, the path above will be returned twice
    3. `**` will match any matched file, directory, symlink and '' by default, when recursive is `True`
    4. fs_glob returns same as glob.glob(pathname, recursive=True) in acsending alphabetical order.
    5. Hidden files (filename stars with '.') will not be found in the result

    :param recursive: If False, `**` will not search directory recursively
    :param missing_ok: If False and target path doesn't match any file, raise FileNotFoundError
    :returns: A list contains paths match `pathname`
    """
    return list(fs_iglob(path=path, recursive=recursive, missing_ok=missing_ok))


def fs_glob_stat(path: PathLike, recursive: bool = True, missing_ok: bool = True) -> Iterator[FileEntry]:
    """Return a list contains tuples of path and file stat, in ascending alphabetical order, in which path matches glob pattern

    1. If doesn't match any path, return empty list
        Notice:  ``glob.glob`` in standard library returns ['a/'] instead of empty list when pathname is like `a/**`, recursive is True and directory 'a' doesn't exist. fs_glob behaves like ``glob.glob`` in standard library under such circumstance.
    2. No guarantee that each path in result is different, which means:
        Assume there exists a path `/a/b/c/b/d.tmp.txt`
        use path pattern like `/**/b/**/*.tmp.txt` to glob, the path above will be returned twice
    3. `**` will match any matched file, directory, symlink and '' by default, when recursive is `True`
    4. fs_glob returns same as glob.glob(pathname, recursive=True) in acsending alphabetical order.
    5. Hidden files (filename stars with '.') will not be found in the result

    :param recursive: If False, `**` will not search directory recursively
    :param missing_ok: If False and target path doesn't match any file, raise FileNotFoundError
    :returns: A list contains tuples of path and file stat, in which paths match `pathname`
    """
    for path in fs_iglob(path=path, recursive=recursive, missing_ok=missing_ok):
        yield FileEntry(os.path.basename(path), path, _make_stat(os.lstat(path)))


def fs_rename(src_path: PathLike, dst_path: PathLike) -> None:
    """
    rename file on fs

    :param src_path: Given path
    :param dst_path: Given destination path
    """
    shutil.move(src_path, dst_path)


def fs_move(src_path: PathLike, dst_path: PathLike) -> None:
    """
    rename file on fs

    :param src_path: Given path
    :param dst_path: Given destination path
    """
    fs_rename(src_path, dst_path)


def fs_resolve(path: PathLike) -> str:
    """Equal to fs_realpath, return the real path of given path

    :param path: Given path
    :returns: Real path of given path
    """
    return FSPath(path).realpath()


def fs_makedirs(path: PathLike, exist_ok: bool = False):
    """
    make a directory on fs, including parent directory

    If there exists a file on the path, raise FileExistsError

    :param path: Given path
    :param exist_ok: If False and target directory exists, raise FileExistsError
    :raises: FileExistsError
    """
    return FSPath(path).mkdir(parents=True, exist_ok=exist_ok)


def fs_lstat(path: PathLike) -> StatResult:
    """
    Like Path.stat() but, if the path points to a symbolic link, return the symbolic link’s information rather than its target’s.

    :param path: Given path
    :returns: StatResult
    """
    return FSPath(path).lstat()


@SmartPath.register
class FSPath(URIPath):
    """file protocol
    e.g. file:///data/test/ or /data/test
    """

    protocol = "file"

    def __init__(self, path: Union["PathLike", int], *other_paths: "PathLike"):
        if not isinstance(path, int):
            if len(other_paths) > 0:
                path = self.from_path(path).joinpath(*other_paths)
            path = str(path)
        self.path = path

    def __fspath__(self) -> str:
        return os.path.normpath(self.path_without_protocol)

    @cachedproperty
    def root(self) -> str:
        return pathlib.Path(self.path_without_protocol).root

    @cachedproperty
    def anchor(self) -> str:
        return pathlib.Path(self.path_without_protocol).anchor

    @cachedproperty
    def drive(self) -> str:
        return pathlib.Path(self.path_without_protocol).drive

    @classmethod
    def from_uri(cls, path: str) -> "FSPath":
        return cls.from_path(path)

    @property
    def path_with_protocol(self) -> Union[str, int]:
        if isinstance(self.path, int):
            return self.path
        protocol_prefix = self.protocol + "://"
        if self.path.startswith(protocol_prefix):
            return self.path
        return protocol_prefix + self.path

    def is_absolute(self) -> bool:
        """Test whether a path is absolute

        :returns: True if a path is absolute, else False
        """
        return os.path.isabs(self.path_without_protocol)

    def abspath(self) -> str:
        """Return the absolute path of given path

        :returns: Absolute path of given path
        """
        return fspath(os.path.abspath(self.path_without_protocol))

    def access(self, mode: Access = Access.READ) -> bool:
        """
        Test if path has access permission described by mode
        Using ``os.access``

        :param mode: access mode
        :returns: Access: Enum, the read/write access that path has.
        """
        if not isinstance(mode, Access):
            raise TypeError(
                "Unsupported mode: {} -- Mode should use one of the enums belonging to:  {}".format(
                    mode, ", ".join([str(a) for a in Access])
                )
            )
        if mode == Access.READ:
            return os.access(self.path_without_protocol, os.R_OK)
        if mode == Access.WRITE:
            return os.access(self.path_without_protocol, os.W_OK)

    def exists(self, followlinks: bool = False) -> bool:
        """
        Test if the path exists

        .. note::

            The difference between this function and ``os.path.exists`` is that this function regard symlink as file.
            In other words, this function is equal to ``os.path.lexists``

        :param followlinks: False if regard symlink as file, else True
        :returns: True if the path exists, else False

        """
        if followlinks:
            return os.path.exists(self.path_without_protocol)
        return os.path.lexists(self.path_without_protocol)

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

    def glob(self, pattern, recursive: bool = True, missing_ok: bool = True) -> List["FSPath"]:
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
        :returns: A list contains tuples of path and file stat, in which paths match `pathname`
        """
        for path_obj in self.iglob(pattern=pattern, recursive=recursive, missing_ok=missing_ok):
            yield FileEntry(path_obj.name, path_obj.path, _make_stat(os.lstat(path_obj.path)))

    def expanduser(self):
        """Expand ~ and ~user constructions.  If user or $HOME is unknown,
        do nothing.
        """
        return os.path.expanduser(self.path_without_protocol)

    def iglob(self, pattern, recursive: bool = True, missing_ok: bool = True) -> Iterator["FSPath"]:
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
        glob_path = self.path_without_protocol
        if pattern:
            glob_path = self.joinpath(pattern).path_without_protocol
        for path in fs_iglob(glob_path, recursive=recursive, missing_ok=missing_ok):
            yield self.from_path(path)

    def is_dir(self, followlinks: bool = False) -> bool:
        """
        Test if a path is directory

        .. note::

            The difference between this function and ``os.path.isdir`` is that this function regard symlink as file

        :param followlinks: False if regard symlink as file, else True
        :returns: True if the path is a directory, else False

        """
        if os.path.islink(self.path_without_protocol) and not followlinks:
            return False
        return os.path.isdir(self.path_without_protocol)

    def is_file(self, followlinks: bool = False) -> bool:
        """
        Test if a path is file

        .. note::

            The difference between this function and ``os.path.isfile`` is that this function regard symlink as file

        :param followlinks: False if regard symlink as file, else True
        :returns: True if the path is a file, else False

        """
        if os.path.islink(self.path_without_protocol) and not followlinks:
            return True
        return os.path.isfile(self.path_without_protocol)

    def listdir(self) -> List[str]:
        """
        Get all contents of given fs path. The result is in acsending alphabetical order.

        :returns: All contents have in the path in acsending alphabetical order
        """
        return sorted(os.listdir(self.path_without_protocol))

    def iterdir(self) -> Iterator["FSPath"]:
        """
        Get all contents of given fs path. The result is in acsending alphabetical order.

        :returns: All contents have in the path in acsending alphabetical order
        """
        for path in self.listdir():
            yield self.joinpath(path)  # type: ignore

    def load(self) -> BinaryIO:
        """Read all content on specified path and write into memory

        User should close the BinaryIO manually

        :returns: Binary stream
        """
        with open(self.path_without_protocol, "rb") as f:
            data = f.read()
        return io.BytesIO(data)

    def mkdir(self, mode=0o777, parents: bool = False, exist_ok: bool = False):
        """
        make a directory on fs, including parent directory

        If there exists a file on the path, raise FileExistsError

        :param mode: If mode is given, it is combined with the process’ umask value to determine the file mode and access flags.
        :param parents: If parents is true, any missing parents of this path are created as needed;
        If parents is false (the default), a missing parent raises FileNotFoundError.
        :param exist_ok: If False and target directory exists, raise FileExistsError
        :raises: FileExistsError
        """
        if exist_ok and self.path_without_protocol == "":
            return
        return pathlib.Path(self.path_without_protocol).mkdir(mode=mode, parents=parents, exist_ok=exist_ok)

    def realpath(self) -> str:
        """Return the real path of given path

        :returns: Real path of given path
        """
        return fspath(os.path.realpath(self.path_without_protocol))

    def relpath(self, start: Optional[str] = None) -> str:
        """Return the relative path of given path

        :param start: Given start directory
        :returns: Relative path from start
        """
        return fspath(os.path.relpath(self.path_without_protocol, start=start))

    def rename(self, dst_path: PathLike) -> "FSPath":
        """
        rename file on fs

        :param dst_path: Given destination path
        """
        fs_rename(self.path_without_protocol, dst_path)
        return self.from_path(dst_path)

    def replace(self, dst_path: PathLike) -> "FSPath":
        """
        move file on fs

        :param dst_path: Given destination path
        """
        return self.rename(dst_path=dst_path)

    def remove(self, missing_ok: bool = False) -> None:
        """
        Remove the file or directory on fs

        :param missing_ok: if False and target file/directory not exists, raise FileNotFoundError
        """
        if missing_ok and not self.exists():
            return
        if self.is_dir():
            shutil.rmtree(self.path_without_protocol)
        else:
            os.remove(self.path_without_protocol)

    def _scan(self, missing_ok: bool = True, followlinks: bool = False) -> Iterator[str]:
        if self.is_file(followlinks=followlinks):
            path = fspath(self.path_without_protocol)
            yield path

        for root, _, files in self.walk(followlinks=followlinks):
            for filename in files:
                yield os.path.join(root, filename)

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
        return _create_missing_ok_generator(
            self._scan(followlinks=followlinks), missing_ok, FileNotFoundError("No match any file in: %r" % self.path)
        )

    def scan_stat(self, missing_ok: bool = True, followlinks: bool = False) -> Iterator[FileEntry]:
        """
        Iteratively traverse only files in given directory, in alphabetical order.
        Every iteration on generator yields a tuple of path string and file stat

        :param missing_ok: If False and there's no file in the directory, raise FileNotFoundError
        :returns: A file path generator
        """
        file_not_found = True
        for path in self._scan(followlinks=followlinks):
            yield FileEntry(os.path.basename(path), path, _make_stat(os.lstat(path)))
            file_not_found = False
        if file_not_found:
            if missing_ok:
                return
            raise FileNotFoundError("No match any file in: %r" % self.path_without_protocol)

    def scandir(self) -> Iterator[FileEntry]:
        """
        Get all content of given file path.

        :returns: An iterator contains all contents have prefix path
        """

        def create_generator():
            with os.scandir(self.path_without_protocol) as entries:
                for entry in entries:
                    yield FileEntry(entry.name, entry.path, _make_stat(entry.stat(follow_symlinks=False)))

        return ContextIterator(create_generator())

    def stat(self, follow_symlinks=True) -> StatResult:
        """
        Get StatResult of file on fs, including file size and mtime, referring to fs_getsize and fs_getmtime

        :returns: StatResult
        """
        if follow_symlinks:
            result = _make_stat(os.stat(self.path_without_protocol))
        else:
            result = _make_stat(os.lstat(self.path_without_protocol))

        if result.islnk or not result.isdir:
            return result

        size = 0
        ctime = result.ctime
        mtime = result.mtime
        for root, _, files in os.walk(self.path_without_protocol):
            for filename in files:
                canonical_path = os.path.join(root, filename)
                stat = os.lstat(canonical_path)
                size += stat.st_size
                if ctime > stat.st_ctime:
                    ctime = stat.st_ctime
                if mtime < stat.st_mtime:
                    mtime = stat.st_mtime
        return result._replace(size=size, ctime=ctime, mtime=mtime)

    def unlink(self, missing_ok: bool = False) -> None:
        """
        Remove the file on fs

        :param missing_ok: if False and target file not exists, raise FileNotFoundError
        """
        if missing_ok and not self.exists():
            return
        os.unlink(self.path_without_protocol)

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

        path = fspath(self.path_without_protocol)
        path = os.path.normpath(self.path_without_protocol)

        stack = [path]
        while stack:
            root = stack.pop()
            dirs, files = [], []
            for entry in os.scandir(root):
                name = fspath(entry.name)
                path = entry.path
                if FSPath(path).is_file(followlinks=followlinks):
                    files.append(name)
                elif FSPath(path).is_dir(followlinks=followlinks):
                    dirs.append(name)

            dirs = sorted(dirs)
            files = sorted(files)

            yield root, dirs, files

            stack.extend((os.path.join(root, directory) for directory in reversed(dirs)))

    def resolve(self, strict=False) -> "FSPath":
        """Equal to fs_realpath

        :return: Return the canonical path of the specified filename, eliminating any symbolic links encountered in the path.
        :rtype: FSPath
        """
        return self.from_path(str(pathlib.Path(self.path_without_protocol).resolve(strict=strict)))

    def md5(self, recalculate: bool = False, followlinks: bool = True):
        """
        Calculate the md5 value of the file

        :param recalculate: Ignore this parameter, just for compatibility
        :param followlinks: Ignore this parameter, just for compatibility
        returns: md5 of file
        """
        if os.path.isdir(self.path_without_protocol):
            hash_md5 = hashlib.md5()  # nosec
            for file_name in self.listdir():
                chunk = (
                    FSPath(self.path_without_protocol, file_name).md5(recalculate=recalculate, followlinks=followlinks).encode()
                )
                hash_md5.update(chunk)
            return hash_md5.hexdigest()
        with open(self.path_without_protocol, "rb") as src:  # type: ignore
            md5 = calculate_md5(src)
        return md5

    def _copyfile(self, dst_path: PathLike, callback: Optional[Callable[[int], None]] = None, followlinks: bool = False):

        shutil.copy2(self.path_without_protocol, dst_path, follow_symlinks=followlinks)

        # After python3.8, patch `shutil.copyfile` is not a good way, because `shutil.copy2` will not call it in some cases.
        if callback:
            callback(self.stat(follow_symlinks=followlinks).size)

    def copy(self, dst_path: PathLike, callback: Optional[Callable[[int], None]] = None, followlinks: bool = False):
        """File copy on file system
        Copy content (excluding meta date) of file on `src_path` to `dst_path`. `dst_path` must be a complete file name

        .. note ::

            The differences between this function and shutil.copyfile are:

                1. If parent directory of dst_path doesn't exist, create it

                2. Allow callback function, None by default. callback: Optional[Callable[[int], None]],

            the int data is means the size (in bytes) of the written data that is passed periodically

                3. This function is thread-unsafe

        :param dst_path: Target file path
        :param callback: Called periodically during copy, and the input parameter is the data size (in bytes) of copy since the last call
        :param followlinks: False if regard symlink as file, else True
        """
        try:
            self._copyfile(dst_path, callback=callback, followlinks=followlinks)
        except FileNotFoundError as error:
            # Prevent the dst_path directory from being created when src_path does not exist
            if dst_path == error.filename:
                FSPath(os.path.dirname(dst_path)).mkdir(parents=True, exist_ok=True)
                self._copyfile(dst_path, callback=callback, followlinks=followlinks)
            else:
                raise

    def sync(self, dst_path: PathLike, followlinks: bool = False, force: bool = False) -> None:
        """Force write of everything to disk.

        :param dst_path: Target file path
        :param followlinks: False if regard symlink as file, else True
        :param force: Sync file forcely, do not ignore same files
        """
        if self.is_dir(followlinks=followlinks):

            def ignore_same_file(src: str, names: List[str]) -> List[str]:
                ignore_files = []
                for name in names:
                    dst_obj = self.from_path(dst_path).joinpath(name)
                    if not force and dst_obj.exists() and is_same_file(self.joinpath(name).stat(), dst_obj.stat(), "copy"):
                        ignore_files.append(name)
                return ignore_files

            copytree(self.path_without_protocol, dst_path, ignore=ignore_same_file, dirs_exist_ok=True)
        else:
            self.copy(dst_path, followlinks=followlinks)

    def symlink(self, dst_path: PathLike) -> None:
        """
        Create a symbolic link pointing to src_path named dst_path.

        :param dst_path: Desination path
        """
        return os.symlink(self.path_without_protocol, dst_path)

    def readlink(self) -> "FSPath":
        """
        Return a FSPath instance representing the path to which the symbolic link points.
        :returns: Return a FSPath instance representing the path to which the symbolic link points.
        """
        return self.from_path(fs_readlink(self.path_without_protocol))

    def is_symlink(self) -> bool:
        """Test whether a path is a symbolic link

        :return: If path is a symbolic link return True, else False
        :rtype: bool
        """
        return os.path.islink(self.path_without_protocol)

    def is_mount(self) -> bool:
        """Test whether a path is a mount point

        :returns: True if a path is a mount point, else False
        """
        return os.path.ismount(self.path_without_protocol)

    def cwd(self) -> "FSPath":
        """Return current working directory

        returns: Current working directory
        """
        return self.from_path(fs_cwd())

    def home(self):
        """Return the home directory

        returns: Home directory path
        """
        return self.from_path(fs_home())

    def joinpath(self, *other_paths: PathLike) -> "FSPath":
        path = fspath(self)
        if path == ".":
            path = ""
        return self.from_path(path_join(path, *map(fspath, other_paths)))

    def save(self, file_object: BinaryIO):
        """Write the opened binary stream to path
        If parent directory of path doesn't exist, it will be created.

        :param file_object: stream to be read
        """
        FSPath(os.path.dirname(self.path_without_protocol)).mkdir(parents=True, exist_ok=True)
        with open(self.path_without_protocol, "wb") as output:
            output.write(file_object.read())

    def open(
        self, mode: str = "r", buffering=-1, encoding=None, errors=None, newline=None, closefd=True, **kwargs
    ) -> IO[AnyStr]:  # pytype: disable=signature-mismatch
        if not isinstance(self.path_without_protocol, int) and ("w" in mode or "x" in mode or "a" in mode):
            FSPath(os.path.dirname(self.path_without_protocol)).mkdir(parents=True, exist_ok=True)
        return io.open(
            self.path_without_protocol,
            mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
            closefd=closefd,
        )

    @cachedproperty
    def parts(self) -> Tuple[str]:
        """
        A tuple giving access to the path’s various components
        """
        return pathlib.Path(self.path_without_protocol).parts

    def chmod(self, mode: int, *, follow_symlinks: bool = True):
        """
        Change the file mode and permissions, like os.chmod().

        This method normally follows symlinks.
        Some Unix flavours support changing permissions on the symlink itself;
        on these platforms you may add the argument follow_symlinks=False, or use lchmod().
        """
        return os.chmod(path=self.path_without_protocol, mode=mode, follow_symlinks=follow_symlinks)

    def group(self) -> str:
        """
        Return the name of the group owning the file. KeyError is raised if the file’s gid isn’t found in the system database.
        """
        return pathlib.Path(self.path_without_protocol).group()

    def is_socket(self) -> bool:
        """
        Return True if the path points to a Unix socket (or a symbolic link pointing to a Unix socket), False if it points to another kind of file.

        False is also returned if the path doesn’t exist or is a broken symlink; other errors (such as permission errors) are propagated.
        """
        return pathlib.Path(self.path_without_protocol).is_socket()

    def is_fifo(self) -> bool:
        """
        Return True if the path points to a FIFO (or a symbolic link pointing to a FIFO), False if it points to another kind of file.

        False is also returned if the path doesn’t exist or is a broken symlink; other errors (such as permission errors) are propagated.
        """
        return pathlib.Path(self.path_without_protocol).is_fifo()

    def is_block_device(self) -> bool:
        """
        Return True if the path points to a block device (or a symbolic link pointing to a block device), False if it points to another kind of file.

        False is also returned if the path doesn’t exist or is a broken symlink; other errors (such as permission errors) are propagated.
        """
        return pathlib.Path(self.path_without_protocol).is_block_device()

    def is_char_device(self) -> bool:
        """
        Return True if the path points to a character device (or a symbolic link pointing to a character device), False if it points to another kind of file.

        False is also returned if the path doesn’t exist or is a broken symlink; other errors (such as permission errors) are propagated.
        """
        return pathlib.Path(self.path_without_protocol).is_char_device()

    def owner(self) -> str:
        """
        Return the name of the user owning the file. KeyError is raised if the file’s uid isn’t found in the system database.
        """
        return pathlib.Path(self.path_without_protocol).owner()

    def absolute(self) -> "FSPath":
        """
        Make the path absolute, without normalization or resolving symlinks. Returns a new path object
        """
        return self.from_path(os.path.abspath(self.path_without_protocol))

    def rmdir(self):
        """
        Remove this directory. The directory must be empty.
        """
        return os.rmdir(self.path_without_protocol)

    def hardlink_to(self, target):
        """
        Make this path a hard link to the same file as target.
        """
        return os.link(target, self.path)

    def utime(self, atime: Union[float, int], mtime: Union[float, int]):
        """
        Set the access and modified times of the file specified by path.

        :param atime: a float or int representing the access time to be set. If it is set to None, the
                      access time is set to the current time.
        :param mtime: a float or int representing the modified time to be set. If it is set to None, the
                      modified time is set to the current time.
        :return: None
        """
        return os.utime(self.path_without_protocol, times=(atime, mtime))
