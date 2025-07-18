# Standard Library
import os
import re
from collections import OrderedDict
from collections import namedtuple as NamedTuple
from typing import Iterator, List, Tuple

# Import from alt
from alt.files.lib import fnmatch

# Python 3.5+ Compatible
"""
class FSFunc(NamedTuple):
    exists: Callable[[str], bool]
    isdir: Callable[[str], bool]
    scandir: Callable[[str], Iterator[Tuple[str, bool]]]  # name, isdir

in Python 3.6+
"""

FSFunc = NamedTuple("FSFunc", ["exists", "isdir", "scandir"])


def _exists(path: str) -> bool:
    return os.path.lexists(path)


def _isdir(path: str) -> bool:
    return os.path.isdir(path)


def _scandir(dirname: str) -> Iterator[Tuple[str, bool]]:
    for entry in sorted(list(os.scandir(dirname)), key=lambda t: t.name):
        yield entry.name, entry.is_dir()


DEFAULT_FILESYSTEM_FUNC = FSFunc(_exists, _isdir, _scandir)


def glob(pathname: str, *, recursive: bool = False, fs: FSFunc = DEFAULT_FILESYSTEM_FUNC) -> List[str]:
    """Return a list of paths matching a pathname pattern.

    The pattern may contain simple shell-style wildcards a la
    fnmatch. However, unlike fnmatch, filenames starting with a
    dot are special cases that are not matched by '*' and '?'
    patterns.

    If recursive is true, the pattern '**' will match any files and
    zero or more directories and subdirectories.
    """
    return list(iglob(pathname, recursive=recursive, fs=fs))


def iglob(pathname: str, *, recursive: bool = False, fs: FSFunc = DEFAULT_FILESYSTEM_FUNC) -> Iterator[str]:
    """Return an iterator which yields the paths matching a pathname pattern.

    The pattern may contain simple shell-style wildcards a la
    fnmatch. However, unlike fnmatch, filenames starting with a
    dot are special cases that are not matched by '*' and '?'
    patterns.

    If recursive is true, the pattern '**' will match any files and
    zero or more directories and subdirectories.
    """
    it = _iglob(pathname, recursive, False, fs)
    if recursive and _isrecursive(pathname):
        s = next(it)  # skip empty string
        assert not s
    return it


def _iglob(pathname: str, recursive: bool, dironly: bool, fs: FSFunc) -> Iterator[str]:
    if "://" in pathname:
        protocol, path_without_protocol = pathname.split("://", 1)
    else:
        protocol, path_without_protocol = "", pathname
    dirname, basename = os.path.split(path_without_protocol)
    if protocol:
        dirname = "://".join([protocol, dirname])
    if not has_magic(pathname):
        assert not dironly
        if basename:
            if fs.exists(pathname):
                yield pathname
        else:
            # Patterns ending with a slash should match only directories
            if fs.isdir(dirname):
                yield pathname
        return
    if not dirname:
        if recursive and _isrecursive(basename):
            yield from _glob2(dirname, basename, dironly, fs)
        else:
            yield from _glob1(dirname, basename, dironly, fs)
        return
    # `os.path.split()` returns the argument itself as a dirname if it is a
    # drive or UNC path.  Prevent an infinite recursion if a drive or UNC path
    # contains magic characters (i.e. r'\\?\C:').
    if dirname != pathname and has_magic(dirname):
        dirs = _iglob(dirname, recursive, True, fs)
    elif fs.exists(dirname):
        dirs = [dirname]
    else:
        dirs = []
    if has_magic(basename):
        if recursive and _isrecursive(basename):
            glob_in_dir = _glob2
        else:
            glob_in_dir = _glob1
    else:
        glob_in_dir = _glob0
    for dirname in dirs:
        for name in glob_in_dir(dirname, basename, dironly, fs):
            yield os.path.join(dirname, name)


# These 2 helper functions non-recursively glob inside a literal directory.
# They return a list of basenames.  _glob1 accepts a pattern while _glob0
# takes a literal basename (so it only has to check for its existence).
def _glob1(dirname: str, pattern: str, dironly: bool, fs: FSFunc) -> List[str]:
    names = list(_iterdir(dirname, dironly, fs))
    if not _ishidden(pattern):
        names = (x for x in names if not _ishidden(x))
    return fnmatch.filter(names, pattern)


def _glob0(dirname: str, basename: str, dironly: bool, fs: FSFunc) -> List[str]:
    if not basename:
        # `os.path.split()` returns an empty basename for paths ending with a
        # directory separator.  'q*x/' should match only directories.
        if fs.isdir(dirname):
            return [basename]
    else:
        if fs.exists(os.path.join(dirname, basename)):
            return [basename]
    return []


# This helper function recursively yields relative pathnames inside a literal
# directory.
def _glob2(dirname: str, pattern: str, dironly: bool, fs: FSFunc) -> Iterator[str]:
    assert _isrecursive(pattern)
    yield pattern[:0]
    yield from _rlistdir(dirname, dironly, fs)


# If dironly is false, yields all file names inside a directory.
# If dironly is true, yields only directory names.
def _iterdir(dirname: str, dironly: bool, fs: FSFunc) -> Iterator[str]:
    if not dirname:
        dirname = os.curdir
    try:
        # dirname may be non-existent, raise OSError
        for name, isdir in fs.scandir(dirname):
            try:
                if not dironly or isdir:
                    yield name  # type: ignore
            except OSError:
                pass
    except OSError:
        return


# Recursively yields relative pathnames inside a literal directory.
def _rlistdir(dirname: str, dironly: bool, fs: FSFunc) -> Iterator[str]:
    names = OrderedDict()
    for name in _iterdir(dirname, dironly, fs):
        names.setdefault(name, 0)
        names[name] += 1
    for x, c in names.items():
        if not _ishidden(x):
            for _ in range(c):
                yield x
            path = os.path.join(dirname, x) if dirname else x
            for y in _rlistdir(path, dironly, fs):
                yield os.path.join(x, y)


magic_check = re.compile(r"([*?[{])")
magic_decheck = re.compile(r"\[(.)\]")
brace_check = re.compile(r"(\{.*\})")
unbrace_check = re.compile(r"([*?[])")


def has_magic(s: str) -> bool:
    match = magic_check.search(s)
    return match is not None


def has_magic_ignore_brace(s: str) -> bool:
    match = unbrace_check.search(brace_check.sub(r"", s))
    return match is not None


def _ishidden(path: str) -> bool:
    return path[0] == "."


def _isrecursive(pattern: str) -> bool:
    return pattern == "**"


def escape(pathname):
    """Escape all special characters."""
    # Escaping is done by wrapping any of "*?[" between square brackets.
    # Metacharacters do not work in the drive part and shouldn't be escaped.
    drive, pathname = os.path.splitdrive(pathname)
    pathname = magic_check.sub(r"[\1]", pathname)
    return drive + pathname


def unescape(pathname):
    """Unescape all special characters."""
    drive, pathname = os.path.splitdrive(pathname)
    pathname = magic_decheck.sub(r"\1", pathname)
    return drive + pathname


def _find_suffix(path_list: List[str], prefix: str, split_sign: str) -> List[str]:
    suffix = []
    temp_path_list = []
    for path_index in range(0, len(path_list)):
        temp_path_list.append(path_list[path_index][len(prefix) :].split(split_sign))
    i = 0
    while True:
        i = i - 1
        if len(temp_path_list[0]) <= abs(i):
            return suffix
        for path_index in range(1, len(path_list)):
            if len(temp_path_list[path_index]) <= abs(i) or temp_path_list[path_index][i] != temp_path_list[0][i]:
                return suffix
        else:
            suffix.insert(0, temp_path_list[0][i])


def globlize(path_list: List[str]) -> str:
    path_list = sorted(path_list)
    if path_list[0] == path_list[-1]:
        return path_list[0]
    first_path = path_list[0].split("/")
    last_path = path_list[-1].split("/")
    prefix = []

    for i in range(0, min(len(first_path), len(last_path))):
        if first_path[i] == last_path[i]:
            prefix.append(first_path[i])
        else:
            break
    if len(prefix) == 0:
        prefix = ""
    else:
        prefix = "/".join(prefix) + "/"
    suffix = _find_suffix(path_list, prefix, "/")

    if len(suffix) == 0:
        suffix = _find_suffix(path_list, prefix, ".")
        if len(suffix) == 0:
            suffix = ""
        else:
            suffix = "." + ".".join(suffix)
    else:
        suffix = "/" + "/".join(suffix)

    path = []
    for i in path_list:
        if i[len(prefix) : len(i) - len(suffix)] not in path:
            path.append(unescape(i[len(prefix) : len(i) - len(suffix)]))
    return prefix + "{" + ",".join(path) + "}" + suffix


def ungloblize(glob: str) -> List[str]:
    path_list = [glob]
    while True:
        temp_path = path_list[0]
        begin = temp_path.find("{")
        end = temp_path.find("}", begin)
        if end == -1:
            break
        path_list.pop(0)
        subpath_list = temp_path[begin + 1 : end].split(",")
        for subpath in subpath_list:
            path = temp_path[:begin] + escape(subpath) + temp_path[end + 1 :]
            path_list.append(path)
    return path_list


def get_non_glob_dir(glob: str):
    root_dir = []
    if glob.startswith("/"):
        root_dir.append("/")
    for name in glob.split("/"):
        if has_magic(name):
            break
        root_dir.append(name)
    if root_dir:
        root_dir = os.path.join(*root_dir)
    else:
        root_dir = "."
    return root_dir
