from re import split
import torch
import torch.distributed as dist


def pre_process(x):
    world_size = dist.get_world_size()
    cur_rank = dist.get_rank()

    x = torch.chunk(x, world_size, dim=0)[cur_rank]

    return x


def post_process(x):
    # 获取当前进程的世界大小
    world_size = dist.get_world_size()

    # 创建一个列表，用于存储所有进程的输出
    gathered_x = [torch.empty_like(x) for _ in range(world_size)]

    # 收集所有进程的输出
    dist.all_gather(gathered_x, x)

    # 在指定的维度上合并所有进程的输出
    combined_output = torch.cat(gathered_x, dim=0)

    return combined_output  # 返回合并后的输出
