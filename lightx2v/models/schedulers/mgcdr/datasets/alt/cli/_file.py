# Standard Library
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# Import from third library
import click
import logging
from tqdm import tqdm

# Import from alt
from alt.files.config import DEFAULT_BLOCK_SIZE
from alt.files.interfaces import FileEntry
from alt.files.lib.glob import get_non_glob_dir, has_magic
from alt.files.smart import (
    _smart_sync_single_file,
    smart_copy,
    smart_exists,
    smart_getmd5,
    smart_getmtime,
    smart_getsize,
    smart_glob_stat,
    smart_isdir,
    smart_isfile,
    smart_makedirs,
    smart_move,
    smart_open,
    smart_path_join,
    smart_readlink,
    smart_realpath,
    smart_remove,
    smart_rename,
    smart_scan_stat,
    smart_scandir,
    smart_stat,
    smart_sync,
    smart_sync_with_progress,
    smart_touch,
    smart_unlink,
)
from alt.files.smart_path import SmartPath
from alt.files.utils import get_human_size

logging.basicConfig(level=logging.ERROR)
logging.getLogger("alt.files").setLevel(level=logging.INFO)
DEBUG = False


@click.group()
@click.pass_context
def cli(ctx):
    """Provide the client about autolabel file client"""
    pass


def safe_cli():  # pragma: no cover
    try:
        cli()
    except Exception as e:
        if DEBUG:
            raise
        else:
            click.echo(f"\n[{type(e).__name__}] {e}", err=True)
            sys.exit(1)


def get_echo_path(file_stat, base_path: str = "", full_path: bool = False):
    if base_path == file_stat.path:
        path = file_stat.name
    elif full_path:
        path = file_stat.path
    else:
        path = os.path.relpath(file_stat.path, start=base_path)
    return path


def simple_echo(file_stat, base_path: str = "", full_path: bool = False):
    return get_echo_path(file_stat, base_path, full_path)


def long_echo(file_stat, base_path: str = "", full_path: bool = False):
    return "%12d %s %s" % (
        file_stat.stat.size,
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(file_stat.stat.mtime)),
        get_echo_path(file_stat, base_path, full_path),
    )


def human_echo(file_stat, base_path: str = "", full_path: bool = False):
    return "%10s %s %s" % (
        get_human_size(file_stat.stat.size),
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(file_stat.stat.mtime)),
        get_echo_path(file_stat, base_path, full_path),
    )


def smart_list_stat(path):
    if smart_isfile(path):
        yield FileEntry(os.path.basename(path), path, smart_stat(path))
    else:
        yield from smart_scandir(path)


def _ls(path: str, long: bool, recursive: bool, human_readable: bool):
    base_path = path
    full_path = False
    if has_magic(path):
        scan_func = smart_glob_stat
        base_path = get_non_glob_dir(path)
        full_path = True
    elif recursive:
        scan_func = smart_scan_stat
    else:
        scan_func = smart_list_stat
    if long:
        if human_readable:
            echo_func = human_echo
        else:
            echo_func = long_echo
    else:
        echo_func = simple_echo

    total_size = 0
    total_count = 0
    for file_stat in scan_func(path):
        total_size += file_stat.stat.size
        total_count += 1
        output = echo_func(file_stat, base_path, full_path=full_path)
        if file_stat.is_symlink():
            try:
                link = smart_readlink(file_stat.path)
            except FileNotFoundError as e:
                link = repr(e)
            output += " -> %s" % link
        click.echo(output)
    if long:
        click.echo(f"total({total_count}): {get_human_size(total_size)}")


@cli.command(short_help="List all the objects in the path.")
@click.argument("path")
@click.option("-l", "--long", is_flag=True, help="List all the objects in the path with size, modification time and path.")
@click.option(
    "-r",
    "--recursive",
    is_flag=True,
    help="Command is performed on all files or objects under the specified directory or prefix.",
)
@click.option("-h", "--human-readable", is_flag=True, help="Displays file sizes in human readable format.")
def ls(path: str, long: bool, recursive: bool, human_readable: bool):
    _ls(path, long=long, recursive=recursive, human_readable=human_readable)


@cli.command(short_help="List all the objects in the path.")
@click.argument("path")
@click.option(
    "-r",
    "--recursive",
    is_flag=True,
    help="Command is performed on all files or objects under the specified directory or prefix.",
)
def ll(path: str, recursive: bool):
    _ls(path, long=True, recursive=recursive, human_readable=True)


@cli.command(short_help="Copy files from source to dest, skipping already copied.")
@click.argument("src_path")
@click.argument("dst_path")
@click.option(
    "-r",
    "--recursive",
    is_flag=True,
    help="Command is performed on all files or objects under the specified directory or prefix.",
)
@click.option("-T", "--no-target-directory", is_flag=True, help="treat dst_path as a normal file.")
@click.option("-g", "--progress-bar", is_flag=True, help="Show progress bar.")
def cp(
    src_path: str,
    dst_path: str,
    recursive: bool,
    no_target_directory: bool,
    progress_bar: bool,
):
    if not no_target_directory and (dst_path.endswith("/") or smart_isdir(dst_path)):
        dst_path = smart_path_join(dst_path, os.path.basename(src_path))

    if recursive:
        with ThreadPoolExecutor(max_workers=(os.cpu_count() or 1) * 2) as executor:
            if progress_bar:
                smart_sync_with_progress(src_path, dst_path, followlinks=True, map_func=executor.map, force=True)
            else:
                smart_sync(src_path, dst_path, followlinks=True, map_func=executor.map, force=True)
    else:
        if progress_bar:
            file_size = smart_stat(src_path).size
            sbar = tqdm(total=file_size, unit="B", ascii=True, unit_scale=True, unit_divisor=1024)

            def callback(length: int):
                sbar.update(length)

            smart_copy(src_path, dst_path, callback=callback)
            sbar.close()
        else:
            smart_copy(src_path, dst_path)


@cli.command(short_help="Move files from source to dest.")
@click.argument("src_path")
@click.argument("dst_path")
@click.option(
    "-r",
    "--recursive",
    is_flag=True,
    help="Command is performed on all files or objects under the specified directory or prefix.",
)
@click.option("-T", "--no-target-directory", is_flag=True, help="treat dst_path as a normal file.")
@click.option("-g", "--progress-bar", is_flag=True, help="Show progress bar.")
def mv(
    src_path: str,
    dst_path: str,
    recursive: bool,
    no_target_directory: bool,
    progress_bar: bool,
):
    if not no_target_directory and (dst_path.endswith("/") or smart_isdir(dst_path)):
        dst_path = smart_path_join(dst_path, os.path.basename(src_path))
    if progress_bar:
        src_protocol, _ = SmartPath._extract_protocol(src_path)
        dst_protocol, _ = SmartPath._extract_protocol(dst_path)

        if recursive:
            if src_protocol == dst_protocol:
                with tqdm(total=1) as t:
                    SmartPath(src_path).rename(dst_path)
                    t.update(1)
            else:
                smart_sync_with_progress(src_path, dst_path, followlinks=True)
                smart_remove(src_path)
        else:
            if src_protocol == dst_protocol:
                with tqdm(total=1) as t:
                    SmartPath(src_path).rename(dst_path)
                    t.update(1)
            else:
                file_size = smart_stat(src_path).size
                sbar = tqdm(total=file_size, unit="B", ascii=True, unit_scale=True, unit_divisor=1024)

                def callback(length: int):
                    sbar.update(length)

                smart_copy(src_path, dst_path, callback=callback)
                smart_unlink(src_path)
                sbar.close()
    else:
        move_func = smart_move if recursive else smart_rename
        move_func(src_path, dst_path)


@cli.command(short_help="Remove files from path.")
@click.argument("path")
@click.option(
    "-r",
    "--recursive",
    is_flag=True,
    help="Command is performed on all files or objects under the specified directory or prefix.",
)
def rm(path: str, recursive: bool):
    remove_func = smart_remove if recursive else smart_unlink
    remove_func(path)


@cli.command(short_help="Make source and dest identical, modifying destination only.")
@click.argument("src_path")
@click.argument("dst_path")
@click.option("-g", "--progress-bar", is_flag=True, help="Show progress bar.")
@click.option("-w", "--worker", type=click.INT, default=8, help="Number of concurrent workers.")
@click.option("-f", "--force", is_flag=True, help="Copy files forcely, ignore same files.")
@click.option("-q", "--quiet", is_flag=True, help="Not show any progress log.")
def sync(src_path: str, dst_path: str, progress_bar: bool, worker: int, force: bool, quiet: bool):
    with ThreadPoolExecutor(max_workers=worker) as executor:
        if has_magic(src_path):
            src_root_path = get_non_glob_dir(src_path)
            if not smart_exists(src_root_path):
                raise FileNotFoundError(f"No match file: {src_path}")

            def scan_func(path):
                for glob_file_entry in smart_glob_stat(path):
                    if glob_file_entry.is_file():
                        yield glob_file_entry
                    else:
                        for file_entry in smart_scan_stat(glob_file_entry.path, followlinks=True):
                            yield file_entry

        else:
            if not smart_exists(src_path):
                raise FileNotFoundError(f"No match file: {src_path}")
            src_root_path = src_path
            scan_func = partial(smart_scan_stat, followlinks=True)

        if progress_bar and not quiet:
            print("building progress bar", end="\r")
            file_entries = []
            total_count = total_size = 0
            for total_count, file_entry in enumerate(scan_func(src_path), start=1):
                if total_count > 1024 * 128:
                    file_entries = []
                else:
                    file_entries.append(file_entry)
                total_size += file_entry.stat.size
                print(f"building progress bar, find {total_count} files", end="\r")

            if not file_entries:
                file_entries = scan_func(src_path)
        else:
            total_count = total_size = None
            file_entries = scan_func(src_path)

        if quiet:
            callback = callback_after_copy_file = None
        else:
            tbar = tqdm(total=total_count, ascii=True)
            sbar = tqdm(unit="B", ascii=True, unit_scale=True, unit_divisor=1024, total=total_size)

            def callback(_filename: str, length: int):
                sbar.update(length)

            def callback_after_copy_file(src_file_path, dst_file_path):
                tbar.update(1)

        params_iter = (
            dict(
                src_root_path=src_root_path,
                dst_root_path=dst_path,
                src_file_path=file_entry.path,
                callback=callback,
                followlinks=True,
                callback_after_copy_file=callback_after_copy_file,
                force=force,
            )
            for file_entry in file_entries
        )
        list(executor.map(_smart_sync_single_file, params_iter))
    if not quiet:
        tbar.close()
        if progress_bar:
            sbar.update(sbar.total - sbar.n)
        sbar.close()


@cli.command(short_help="Make the path if it doesn't already exist.")
@click.argument("path")
def mkdir(path: str):
    smart_makedirs(path)


@cli.command(short_help="Make the file if it doesn't already exist.")
@click.argument("path")
def touch(path: str):
    smart_touch(path)


@cli.command(short_help="Concatenate any files and send them to stdout.")
@click.argument("path")
def cat(path: str):
    with smart_open(path, "rb") as file:
        shutil.copyfileobj(file, sys.stdout.buffer)


@cli.command(short_help="Concatenate any files and send first n lines of them to stdout.")
@click.argument("path")
@click.option("-n", "--lines", type=click.INT, default=10, help="print the first NUM lines")
def head(path: str, lines: int):
    with smart_open(path, "rb") as f:
        for _ in range(lines):
            try:
                content = f.readline()
                if not content:
                    break
            except EOFError:
                break
            click.echo(content.strip(b"\n"))


@cli.command(short_help="Concatenate any files and send last n lines of them to stdout.")
@click.argument("path")
@click.option("-n", "--lines", type=click.INT, default=10, help="print the last NUM lines")
def tail(path: str, lines: int):
    line_list = []
    with smart_open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(0, os.SEEK_SET)

        for current_offset in range(file_size - DEFAULT_BLOCK_SIZE, 0 - DEFAULT_BLOCK_SIZE, -DEFAULT_BLOCK_SIZE):
            current_offset = max(0, current_offset)
            f.seek(current_offset)
            block_lines = f.read(DEFAULT_BLOCK_SIZE).split(b"\n")
            if len(line_list) > 0:
                block_lines[-1] += line_list[0]
                block_lines.extend(line_list[1:])
            if len(block_lines) > lines:
                line_list = block_lines[-lines:]
                break
            else:
                line_list = block_lines
    for line in line_list:
        click.echo(line)


@cli.command(short_help="Write bytes from stdin to file.")
@click.argument("path")
@click.option("-a", "--append", is_flag=True, help="Append to the given file")
@click.option("-o", "--stdout", is_flag=True, help="File content to standard output")
def to(path: str, append: bool, stdout: bool):
    mode = "wb"
    if append:
        mode = "ab"
    with smart_open("stdio://0", "rb") as stdin, smart_open(path, mode) as f, smart_open("stdio://1", "wb") as stdout_fd:
        length = 16 * 1024
        while True:
            buf = stdin.read(length)
            if not buf:
                break
            f.write(buf)
            if stdout:
                stdout_fd.write(buf)


@cli.command(short_help="Produce an md5sum file for all the objects in the path.")
@click.argument("path")
def md5sum(path: str):
    click.echo(smart_getmd5(path, recalculate=True))


@cli.command(short_help="Return the total size and number of objects in remote:path.")
@click.argument("path")
def size(path: str):
    click.echo(smart_getsize(path))


@cli.command(short_help="Return the mtime and number of objects in remote:path.")
@click.argument("path")
def mtime(path: str):
    click.echo(smart_getmtime(path))


@cli.command(short_help="Return the stat and number of objects in remote:path.")
@click.argument("path")
def stat(path: str):
    click.echo(smart_stat(path))


if __name__ == "__main__":
    cli()
