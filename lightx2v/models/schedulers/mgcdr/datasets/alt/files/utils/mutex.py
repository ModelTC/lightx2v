import os
from abc import ABC, abstractmethod
from functools import wraps
from threading import RLock
from threading import local as _ThreadLocal
from typing import Any, Callable, Iterator

__all__ = [
    'ThreadLocal',
    'ProcessLocal',
]


class ForkAware(ABC):

    def __init__(self):
        self._process_id = os.getpid()
        self._reset()

    def __reduce__(self):
        return type(self), ()

    @abstractmethod
    def _reset(self):
        pass  # pragma: no cover


def fork_aware(func):

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        current_pid = os.getpid()
        if self._process_id != current_pid:
            self._reset()
            self._process_id = current_pid
        return func(self, *args, **kwargs)

    return wrapper


class BaseLocal(ABC):  # pragma: no cover

    @property
    @abstractmethod
    def _data(self) -> dict:
        pass

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __setitem__(self, key: str, value: Any):
        self._data[key] = value

    def __delitem__(self, key: str):
        del self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator:
        return iter(self._data)


class ThreadLocal(ForkAware, BaseLocal):

    def _reset(self):
        self._local = _ThreadLocal()

    @property
    @fork_aware
    def _data(self):
        return self._local.__dict__

    def __call__(self, key: str, func: Callable, *args, **kwargs):
        data = self._data  # 不同线程拿到的 dict 不同, 因此不用加锁
        if key not in data:
            data[key] = func(*args, **kwargs)
        return data[key]


class ProcessLocal(ForkAware, BaseLocal):
    """
    Provides a basic per-process mapping container that wipes itself if the current PID changed since the last get/set.
    Aka `threading.local()`, but for processes instead of threads.
    """
    _lock = None

    def _reset(self):
        self._lock = RLock()
        self._local = {}

    @property
    @fork_aware
    def _data(self):
        return self._local

    def __call__(self, key: str, func: Callable, *args, **kwargs) -> Any:
        with self._lock:
            data = self._data
            if key not in data:
                data[key] = func(*args, **kwargs)
            return data[key]
