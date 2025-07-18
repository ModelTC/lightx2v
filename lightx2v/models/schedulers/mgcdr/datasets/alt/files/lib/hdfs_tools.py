try:
    import hdfs as hdfs_api
except ImportError:  # pragma: no cover
    hdfs_api = None

__all__ = [
    'hdfs_api',
]

if hdfs_api:
    _to_error = hdfs_api.client._to_error

    def _patch_to_error(response):
        try:
            err = _to_error(response)
        except hdfs_api.HdfsError as e:
            err = e
        err.status_code = response.status_code
        return err

    hdfs_api.client._to_error = _patch_to_error
