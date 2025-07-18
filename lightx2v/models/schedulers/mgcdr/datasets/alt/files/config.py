# Standard Library
import os

DEFAULT_BLOCK_SIZE = int(os.getenv("alt.file_BLOCK_SIZE") or 8 * 2**20)  # 8MB
DEFAULT_MAX_BLOCK_SIZE = int(os.getenv("alt.file_MAX_BLOCK_SIZE") or DEFAULT_BLOCK_SIZE * 16)  # 128MB
DEFAULT_MAX_BUFFER_SIZE = int(os.getenv("alt.file_MAX_BUFFER_SIZE") or DEFAULT_BLOCK_SIZE * 16)  # 128MB
GLOBAL_MAX_WORKERS = int(os.getenv("alt.file_MAX_WORKERS") or 32)
DEFAULT_BLOCK_CAPACITY = int(os.getenv("alt.file_BLOCK_CAPACITY") or 16)

# for logging the size of file had read or wrote
BACKOFF_INITIAL = 64 * 2**20  # 64MB
BACKOFF_FACTOR = 4

NEWLINE = ord("\n")

S3_CLIENT_CACHE_MODE = os.getenv("alt.file_S3_CLIENT_CACHE_MODE") or "thread_local"
