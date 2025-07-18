# Standard Library
from os import stat_result
from typing import Union

# Import from alt
from alt.files.pathlike import StatResult


def get_sync_type(src_protocol, dst_protocol):
    if src_protocol == "s3" and dst_protocol != "s3":
        return "download"
    elif src_protocol != "s3" and dst_protocol == "s3":
        return "upload"
    else:
        return "copy"


def compare_time(src_stat: Union[StatResult, stat_result], dest_stat: Union[StatResult, stat_result], sync_type: str):
    """
    :returns: True if the file does not need updating based on time of
        last modification and type of operation.
        False if the file does need updating based on the time of
        last modification and type of operation.
    """
    src_time = src_stat.st_mtime
    dest_time = dest_stat.st_mtime
    delta = dest_time - src_time
    if sync_type == "upload" or sync_type == "copy":
        if delta >= 0:
            # Destination is newer than source.
            return True
        else:
            # Destination is older than source, so
            # we have a more recently updated file
            # at the source location.
            return False
    elif sync_type == "download":

        if delta <= 0:
            return True
        else:
            # delta is positive, so the destination
            # is newer than the source.
            return False


def is_same_file(src_stat: Union[StatResult, stat_result], dest_stat: Union[StatResult, stat_result], sync_type: str):
    """
    Determines whether or not the source and destination files should be synced based on a comparison of their size and last modified time.

    :param src_stat: A Union[StatResult, stat_result] object representing the source file to be compared.
    :type src_stat: Union[StatResult, stat_result]
    :param dest_stat: A Union[StatResult, stat_result] object representing the destination file to be compared.
    :type dest_stat: Union[StatResult, stat_result]

    :return: A boolean value indicating whether or not the files should be synced.
    :rtype: bool
    """
    same_last_modified_time = compare_time(src_stat, dest_stat, sync_type)
    return src_stat.st_size == dest_stat.st_size and same_last_modified_time
