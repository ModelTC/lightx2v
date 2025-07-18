# Standard Library
from pathlib import PurePath
from typing import Tuple, Union

# Import from alt
from alt.files.lib.compat import fspath
from alt.files.lib.url import get_url_scheme

# Import from local
from .errors import ProtocolExistsError, ProtocolNotFoundError
from .interfaces import BasePath, BaseURIPath, PathLike


def _bind_function(name):

    def smart_method(self, *args, **kwargs):
        return getattr(self.pathlike, name)(*args, **kwargs)

    smart_method.__name__ = name

    return smart_method


def _bind_property(name):

    @property
    def smart_property(self):
        return getattr(self.pathlike, name)

    return smart_property


class SmartPath(BasePath):
    _registered_protocols = dict()

    def __init__(self, path: Union[PathLike, int], *other_paths: PathLike):
        self.path = str(path) if not isinstance(path, int) else path
        pathlike = path
        if not isinstance(pathlike, BaseURIPath):
            pathlike = self._create_pathlike(path)
        if len(other_paths) > 0:
            pathlike = pathlike.joinpath(*other_paths)
            self.path = str(pathlike)
        self.pathlike = pathlike

    @staticmethod
    def _extract_protocol(path: Union[PathLike, int]) -> Tuple[str, Union[str, int]]:
        if isinstance(path, int):
            protocol = "file"
            path_without_protocol = path
        elif isinstance(path, str):
            protocol = get_url_scheme(path)
            if protocol.endswith(":s3"):
                protocol = "s3"
            if protocol == "":
                protocol = "file"
                path_without_protocol = path
            else:
                path_without_protocol = path.split("://")[-1]
        elif isinstance(path, (BaseURIPath, SmartPath)):
            protocol = path.protocol
            path_without_protocol = str(path)
        elif isinstance(path, (PurePath, BasePath)):
            protocol, path_without_protocol = SmartPath._extract_protocol(fspath(path))
        else:
            raise ProtocolNotFoundError("protocol not found: %r" % path)
        return protocol, path_without_protocol

    @classmethod
    def _create_pathlike(cls, path: Union[PathLike, int]) -> BasePath:
        protocol, _ = cls._extract_protocol(path)
        if "s3" in protocol:
            protocol = "s3"
        if protocol not in cls._registered_protocols:
            raise ProtocolNotFoundError("protocol %r not found: %r" % (protocol, path))
        path_class = cls._registered_protocols[protocol]
        return path_class(path)

    @classmethod
    def register(cls, path_class, override_ok: bool = False):
        protocol = path_class.protocol
        if protocol in cls._registered_protocols and not override_ok:
            raise ProtocolExistsError("protocol already exists: %r" % protocol)
        cls._registered_protocols[protocol] = path_class
        return path_class

    symlink = _bind_function("symlink")
    symlink_to = _bind_function("symlink_to")
    hardlink_to = _bind_function("hardlink_to")
    readlink = _bind_function("readlink")
    is_dir = _bind_function("is_dir")
    is_file = _bind_function("is_file")
    is_symlink = _bind_function("is_symlink")
    access = _bind_function("access")
    exists = _bind_function("exists")
    listdir = _bind_function("listdir")
    scandir = _bind_function("scandir")
    getsize = _bind_function("getsize")
    getmtime = _bind_function("getmtime")
    stat = _bind_function("stat")
    lstat = _bind_function("lstat")
    remove = _bind_function("remove")
    rename = _bind_function("rename")
    replace = _bind_function("replace")
    unlink = _bind_function("unlink")
    mkdir = _bind_function("mkdir")
    open = _bind_function("open")
    touch = _bind_function("touch")
    walk = _bind_function("walk")
    scan = _bind_function("scan")
    scan_stat = _bind_function("scan_stat")
    glob = _bind_function("glob")
    iglob = _bind_function("iglob")
    glob_stat = _bind_function("glob_stat")
    load = _bind_function("load")
    save = _bind_function("save")
    joinpath = _bind_function("joinpath")
    abspath = _bind_function("abspath")
    realpath = _bind_function("realpath")
    relpath = _bind_function("relpath")
    is_absolute = _bind_function("is_absolute")
    is_mount = _bind_function("is_mount")
    md5 = _bind_function("md5")

    @property
    def protocol(self) -> str:
        return self.pathlike.protocol  # pytype: disable=attribute-error

    @classmethod
    def from_uri(cls, path: str):
        return cls(path)

    as_uri = _bind_function("as_uri")
    as_posix = _bind_function("as_posix")
    __lt__ = _bind_function("__lt__")
    __le__ = _bind_function("__le__")
    __gt__ = _bind_function("__gt__")
    __ge__ = _bind_function("__ge__")
    __fspath__ = _bind_function("__fspath__")
    __truediv__ = _bind_function("__truediv__")

    joinpath = _bind_function("joinpath")
    is_reserved = _bind_function("is_reserved")
    match = _bind_function("match")
    relative_to = _bind_function("relative_to")
    with_name = _bind_function("with_name")
    with_suffix = _bind_function("with_suffix")
    with_stem = _bind_function("with_stem")
    is_absolute = _bind_function("is_absolute")
    is_mount = _bind_function("is_mount")
    abspath = _bind_function("abspath")
    realpath = _bind_function("realpath")
    relpath = _bind_function("relpath")
    iterdir = _bind_function("iterdir")
    cwd = _bind_function("cwd")
    home = _bind_function("home")
    expanduser = _bind_function("expanduser")
    resolve = _bind_function("resolve")
    chmod = _bind_function("chmod")
    lchmod = _bind_function("lchmod")
    group = _bind_function("group")
    is_socket = _bind_function("is_socket")
    is_fifo = _bind_function("is_fifo")
    is_block_device = _bind_function("is_block_device")
    is_char_device = _bind_function("is_char_device")
    owner = _bind_function("owner")
    absolute = _bind_function("absolute")
    rmdir = _bind_function("rmdir")
    is_relative_to = _bind_function("is_relative_to")
    read_bytes = _bind_function("read_bytes")
    read_text = _bind_function("read_text")
    rglob = _bind_function("rglob")
    samefile = _bind_function("samefile")
    write_bytes = _bind_function("write_bytes")
    write_text = _bind_function("write_text")
    utime = _bind_function("utime")

    drive = _bind_property("drive")
    root = _bind_property("root")
    anchor = _bind_property("anchor")
    parts = _bind_property("parts")
    parents = _bind_property("parents")
    parent = _bind_property("parent")
    name = _bind_property("name")
    suffix = _bind_property("suffix")
    suffixes = _bind_property("suffixes")
    stem = _bind_property("stem")


def get_traditional_path(path: str):
    return fspath(SmartPath(path).path)
