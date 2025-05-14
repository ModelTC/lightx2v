import torch
from loguru import logger


def peak_memory_decorator(func):
    def wrapper(*args, **kwargs):
        # 检查是否在分布式环境中
        rank_info = ""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            rank_info = f"Rank {rank} - "

        # 如果使用GPU，重置显存统计
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # 执行目标函数
        result = func(*args, **kwargs)
        
        # 获取峰值显存
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # 转换为GB
            logger.info(f"{rank_info}Function '{func.__qualname__}' Peak Memory: {peak_memory:.2f} GB")
        else:
            logger.info(f"{rank_info}Function '{func.__qualname__}' executed without GPU.")
        
        return result
    
    return wrapper