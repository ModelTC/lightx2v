# Standard Library
import atexit
import concurrent.futures  # noqa don't delete this import, to ensure the _close_s3_pipes registration is earlier than concurrent.futures._python_exit
import os
from threading import Thread
from typing import Optional

# Import from alt
from alt.files.errors import translate_s3_error
from alt.files.interfaces import Readable, Writable

_s3_opened_pipes = []


@atexit.register
def _close_s3_pipes():  # pragma: no cover

    def try_close_pipe(fd):
        try:
            os.close(fd)
        except IOError:
            pass

    for r, w in _s3_opened_pipes:
        try_close_pipe(r)
        try_close_pipe(w)


class S3PipeHandler(Readable, Writable):

    def __init__(
        self, bucket: str, key: str, mode: str, *, s3_client, join_thread: bool = True, profile_name: Optional[str] = None
    ):

        assert mode in ("rb", "wb")

        self._bucket = bucket
        self._key = key
        self._mode = mode
        self._client = s3_client
        self._join_thread = join_thread
        self._offset = 0
        self._profile_name = profile_name

        self._exc = None
        self._pipe = os.pipe()
        _s3_opened_pipes.append(self._pipe)

        if self._mode == "rb":
            self._fileobj = os.fdopen(self._pipe[0], "rb")
            self._async_task = Thread(target=self._download_fileobj, daemon=True)
        else:
            self._fileobj = os.fdopen(self._pipe[1], "wb")
            self._async_task = Thread(target=self._upload_fileobj, daemon=True)
        self._async_task.start()

    @property
    def name(self) -> str:
        return "s3%s://%s/%s" % (f"+{self._profile_name}" if self._profile_name else "", self._bucket, self._key)

    @property
    def mode(self) -> str:
        return self._mode

    def tell(self) -> int:
        return self._offset

    def _download_fileobj(self):
        try:
            with os.fdopen(self._pipe[1], "wb") as buffer:
                self._client.download_fileobj(self._bucket, self._key, buffer)
        except BrokenPipeError:
            if self._fileobj.closed:
                return
            raise
        except Exception as error:
            self._exc = error

    def _upload_fileobj(self):
        try:
            with os.fdopen(self._pipe[0], "rb") as buffer:
                self._client.upload_fileobj(buffer, self._bucket, self._key)
        except Exception as error:
            self._exc = error

    def _raise_exception(self):
        if self._exc is not None:
            raise translate_s3_error(self._exc, self.name)

    def readable(self) -> bool:
        return self._mode == "rb"

    def read(self, size: Optional[int] = None) -> bytes:
        self._raise_exception()
        data = self._fileobj.read(size)
        self._offset += len(data)
        return data

    def readline(self, size: Optional[int] = None) -> bytes:
        self._raise_exception()
        data = self._fileobj.readline(size)
        self._offset += len(data)
        return data

    def writable(self) -> bool:
        return self._mode == "wb"

    def flush(self):
        self._fileobj.flush()

    def write(self, data: bytes) -> int:
        self._raise_exception()
        self._offset += len(data)
        return self._fileobj.write(data)

    def _close(self):
        self._fileobj.close()
        if self._join_thread:
            self._async_task.join()
        _s3_opened_pipes.remove(self._pipe)
        self._raise_exception()
