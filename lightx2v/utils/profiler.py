import time
import torch
from contextlib import ContextDecorator
from lightx2v.utils.envs import *
from loguru import logger


class _ProfilingContext(ContextDecorator):
    def __init__(self, name):
        self.name = name
        self.rank_info = ""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            self.rank_info = f"Rank {rank} - "

    def __enter__(self):
        torch.cuda.synchronize()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # 转换为GB
            logger.info(f"{self.rank_info}Function '{self.name}' Peak Memory: {peak_memory:.2f} GB")
        else:
            logger.info(f"{self.rank_info}Function '{self.name}' executed without GPU.")
        elapsed = time.perf_counter() - self.start_time
        logger.info(f"[Profile] {self.name} cost {elapsed:.6f} seconds")
        return False


class _NullContext(ContextDecorator):
    # Context manager without decision branch logic overhead
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


ProfilingContext = _ProfilingContext
ProfilingContext4Debug = _ProfilingContext if CHECK_ENABLE_PROFILING_DEBUG() else _NullContext
