# Standard Library
from typing import (
    IO,
    AnyStr,
    BinaryIO,
    Callable,
    Iterator,
    List,
    Optional,
    Tuple,
)

# Import from alt
from alt.files.interfaces import FileEntry, PathLike, StatResult
from alt.files.sftp_path import (
    SftpPath,
    is_sftp,
    sftp_concat,
    sftp_download,
    sftp_glob,
    sftp_glob_stat,
    sftp_iglob,
    sftp_lstat,
    sftp_path_join,
    sftp_readlink,
    sftp_resolve,
    sftp_upload,
)

__all__ = [
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
    "sftp_exists",
    "sftp_getmtime",
    "sftp_getsize",
    "sftp_isdir",
    "sftp_isfile",
    "sftp_listdir",
    "sftp_load_from",
    "sftp_makedirs",
    "sftp_realpath",
    "sftp_rename",
    "sftp_move",
    "sftp_remove",
    "sftp_scan",
    "sftp_scan_stat",
    "sftp_scandir",
    "sftp_stat",
    "sftp_unlink",
    "sftp_walk",
    "sftp_getmd5",
    "sftp_symlink",
    "sftp_islink",
    "sftp_save_as",
    "sftp_open",
    "sftp_chmod",
    "sftp_absolute",
    "sftp_rmdir",
    "sftp_copy",
    "sftp_sync",
]


def sftp_exists(path: PathLike, followlinks: bool = False) -> bool:
    """
    Test if the path exists

    :param path: Given path
    :param followlinks: False if regard symlink as file, else True
    :returns: True if the path exists, else False

    """
    return SftpPath(path).exists(followlinks)


def sftp_getmtime(path: PathLike, follow_symlinks: bool = False) -> float:
    """
    Get last-modified time of the file on the given path (in Unix timestamp format).
    If the path is an existent directory, return the latest modified time of all file in it.

    :param path: Given path
    :returns: last-modified time
    """
    return SftpPath(path).getmtime(follow_symlinks)


def sftp_getsize(path: PathLike, follow_symlinks: bool = False) -> int:
    """
    Get file size on the given file path (in bytes).
    If the path in a directory, return the sum of all file size in it, including file in subdirectories (if exist).
    The result excludes the size of directory itself. In other words, return 0 Byte on an empty directory path.

    :param path: Given path
    :returns: File size

    """
    return SftpPath(path).getsize(follow_symlinks)


def sftp_isdir(path: PathLike, followlinks: bool = False) -> bool:
    """
    Test if a path is directory

    .. note::

        The difference between this function and ``os.path.isdir`` is that this function regard symlink as file

    :param path: Given path
    :param followlinks: False if regard symlink as file, else True
    :returns: True if the path is a directory, else False

    """
    return SftpPath(path).is_dir(followlinks)


def sftp_isfile(path: PathLike, followlinks: bool = False) -> bool:
    """
    Test if a path is file

    .. note::

        The difference between this function and ``os.path.isfile`` is that this function regard symlink as file

    :param path: Given path
    :param followlinks: False if regard symlink as file, else True
    :returns: True if the path is a file, else False

    """
    return SftpPath(path).is_file(followlinks)


def sftp_listdir(path: PathLike) -> List[str]:
    """
    Get all contents of given sftp path. The result is in acsending alphabetical order.

    :param path: Given path
    :returns: All contents have in the path in acsending alphabetical order
    """
    return SftpPath(path).listdir()


def sftp_load_from(path: PathLike) -> BinaryIO:
    """Read all content on specified path and write into memory

    User should close the BinaryIO manually

    :param path: Given path
    :returns: Binary stream
    """
    return SftpPath(path).load()


def sftp_makedirs(path: PathLike, mode=0o777, parents: bool = False, exist_ok: bool = False):
    """
    make a directory on sftp, including parent directory

    If there exists a file on the path, raise FileExistsError

    :param path: Given path
    :param mode: If mode is given, it is combined with the process’ umask value to determine the file mode and access flags.
    :param parents: If parents is true, any missing parents of this path are created as needed;
    If parents is false (the default), a missing parent raises FileNotFoundError.
    :param exist_ok: If False and target directory exists, raise FileExistsError
    :raises: FileExistsError
    """
    return SftpPath(path).mkdir(mode, parents, exist_ok)


def sftp_realpath(path: PathLike) -> str:
    """Return the real path of given path

    :param path: Given path
    :returns: Real path of given path
    """
    return SftpPath(path).realpath()


def sftp_rename(src_path: PathLike, dst_path: PathLike) -> "SftpPath":
    """
    rename file on sftp

    :param src_path: Given path
    :param dst_path: Given destination path
    """
    return SftpPath(src_path).rename(dst_path)


def sftp_move(src_path: PathLike, dst_path: PathLike) -> "SftpPath":
    """
    move file on sftp

    :param src_path: Given path
    :param dst_path: Given destination path
    """
    return SftpPath(src_path).replace(dst_path)


def sftp_remove(path: PathLike, missing_ok: bool = False) -> None:
    """
    Remove the file or directory on sftp

    :param path: Given path
    :param missing_ok: if False and target file/directory not exists, raise FileNotFoundError
    """
    return SftpPath(path).remove(missing_ok)


def sftp_scan(path: PathLike, missing_ok: bool = True, followlinks: bool = False) -> Iterator[str]:
    """
    Iteratively traverse only files in given directory, in alphabetical order.
    Every iteration on generator yields a path string.

    If path is a file path, yields the file only
    If path is a non-existent path, return an empty generator
    If path is a bucket path, return all file paths in the bucket

    :param path: Given path
    :param missing_ok: If False and there's no file in the directory, raise FileNotFoundError
    :returns: A file path generator
    """
    return SftpPath(path).scan(missing_ok, followlinks)


def sftp_scan_stat(path: PathLike, missing_ok: bool = True, followlinks: bool = False) -> Iterator[FileEntry]:
    """
    Iteratively traverse only files in given directory, in alphabetical order.
    Every iteration on generator yields a tuple of path string and file stat

    :param path: Given path
    :param missing_ok: If False and there's no file in the directory, raise FileNotFoundError
    :returns: A file path generator
    """
    return SftpPath(path).scan_stat(missing_ok, followlinks)


def sftp_scandir(path: PathLike) -> Iterator[FileEntry]:
    """
    Get all content of given file path.

    :param path: Given path
    :returns: An iterator contains all contents have prefix path
    """
    return SftpPath(path).scandir()


def sftp_stat(path: PathLike, follow_symlinks=True) -> StatResult:
    """
    Get StatResult of file on sftp, including file size and mtime, referring to fs_getsize and fs_getmtime

    :param path: Given path
    :returns: StatResult
    """
    return SftpPath(path).stat(follow_symlinks)


def sftp_unlink(path: PathLike, missing_ok: bool = False) -> None:
    """
    Remove the file on sftp

    :param path: Given path
    :param missing_ok: if False and target file not exists, raise FileNotFoundError
    """
    return SftpPath(path).unlink(missing_ok)


def sftp_walk(path: PathLike, followlinks: bool = False) -> Iterator[Tuple[str, List[str], List[str]]]:
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

    :param path: Given path
    :param followlinks: False if regard symlink as file, else True
    :returns: A 3-tuple generator
    """
    return SftpPath(path).walk(followlinks)


def sftp_getmd5(path: PathLike, recalculate: bool = False, followlinks: bool = True):
    """
    Calculate the md5 value of the file

    :param path: Given path
    :param recalculate: Ignore this parameter, just for compatibility
    :param followlinks: Ignore this parameter, just for compatibility
    returns: md5 of file
    """
    return SftpPath(path).md5(recalculate, followlinks)


def sftp_symlink(src_path: PathLike, dst_path: PathLike) -> None:
    """
    Create a symbolic link pointing to src_path named dst_path.

    :param src_path: Given path
    :param dst_path: Desination path
    """
    return SftpPath(src_path).symlink(dst_path)


def sftp_islink(path: PathLike) -> bool:
    """Test whether a path is a symbolic link

    :param path: Given path
    :return: If path is a symbolic link return True, else False
    :rtype: bool
    """
    return SftpPath(path).is_symlink()


def sftp_save_as(file_object: BinaryIO, path: PathLike):
    """Write the opened binary stream to path
    If parent directory of path doesn't exist, it will be created.

    :param path: Given path
    :param file_object: stream to be read
    """
    return SftpPath(path).save(file_object)


def sftp_open(
    path: PathLike, mode: str = "r", buffering=-1, encoding: Optional[str] = None, errors: Optional[str] = None, **kwargs
) -> IO[AnyStr]:  # pytype: disable=signature-mismatch
    """Open a file on the path.

    :param path: Given path
    :param mode: Mode to open file
    :param buffering: buffering is an optional integer used to set the buffering policy.
    :param encoding: encoding is the name of the encoding used to decode or encode the file. This should only be used in text mode.
    :param errors: errors is an optional string that specifies how encoding and decoding errors are to be handled—this cannot be used in binary mode.
    :returns: File-Like object
    """
    return SftpPath(path).open(mode, buffering, encoding, errors)


def sftp_chmod(path: PathLike, mode: int, follow_symlinks: bool = True):
    """
    Change the file mode and permissions, like os.chmod().

    :param path: Given path
    :param mode: the file mode you want to change
    :param followlinks: Ignore this parameter, just for compatibility
    """
    return SftpPath(path).chmod(mode, follow_symlinks)


def sftp_absolute(path: PathLike) -> "SftpPath":
    """
    Make the path absolute, without normalization or resolving symlinks. Returns a new path object
    """
    return SftpPath(path).absolute()


def sftp_rmdir(path: PathLike):
    """
    Remove this directory. The directory must be empty.
    """
    return SftpPath(path).rmdir()


def sftp_copy(
    src_path: PathLike, dst_path: PathLike, callback: Optional[Callable[[int], None]] = None, followlinks: bool = False
):
    """
    Copy the file to the given destination path.

    :param src_path: Given path
    :param dst_path: The destination path to copy the file to.
    :param callback: An optional callback function that takes an integer parameter and is called
                    periodically during the copy operation to report the number of bytes copied.
    :param followlinks: Whether to follow symbolic links when copying directories.
    :raises IsADirectoryError: If the source is a directory.
    :raises OSError: If there is an error copying the file.
    """
    return SftpPath(src_path).copy(dst_path, callback, followlinks)


def sftp_sync(src_path: PathLike, dst_path: PathLike, followlinks: bool = False, force: bool = False):
    """Copy file/directory on src_url to dst_url

    :param src_path: Given path
    :param dst_url: Given destination path
    :param followlinks: False if regard symlink as file, else True
    :param force: Sync file forcely, do not ignore same files
    """
    return SftpPath(src_path).sync(dst_path, followlinks, force)
