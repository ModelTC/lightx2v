import torch
import torch.distributed as dist


def pre_process(latent_model_input, freqs_cos, freqs_sin):
    """
    对输入的潜在模型数据和频率数据进行预处理，进行切分以适应分布式计算。

    参数:
        latent_model_input (torch.Tensor): 输入的潜在模型数据，形状为 [batch_size, channels, temporal_size, height, width]
        freqs_cos (torch.Tensor): 余弦频率数据，形状为 [batch_size, channels, temporal_size, height, width]
        freqs_sin (torch.Tensor): 正弦频率数据，形状为 [batch_size, channels, temporal_size, height, width]

    返回:
        tuple: 处理后的 latent_model_input, freqs_cos, freqs_sin 和切分维度 split_dim
    """
    # 获取当前进程的世界大小和当前进程的排名
    world_size = dist.get_world_size()
    cur_rank = dist.get_rank()

    # 根据输入的形状确定切分维度
    if latent_model_input.shape[-2] // 2 % world_size == 0:
        split_dim = -2  # 按高度切分
    elif latent_model_input.shape[-1] // 2 % world_size == 0:
        split_dim = -1  # 按宽度切分
    else:
        raise ValueError(f"Cannot split video sequence into world size ({world_size}) parts evenly")

    # 获取时间维度、处理后的高度和宽度
    temporal_size, h, w = latent_model_input.shape[2], latent_model_input.shape[3] // 2, latent_model_input.shape[4] // 2

    # 按照确定的维度切分潜在模型输入
    latent_model_input = torch.chunk(latent_model_input, world_size, dim=split_dim)[cur_rank]

    # 处理余弦频率数据
    dim_thw = freqs_cos.shape[-1]  # 获取频率数据的最后一个维度
    freqs_cos = freqs_cos.reshape(temporal_size, h, w, dim_thw)  # 重塑为 [temporal_size, height, width, dim_thw]
    freqs_cos = torch.chunk(freqs_cos, world_size, dim=split_dim - 1)[cur_rank]  # 切分频率数据
    freqs_cos = freqs_cos.reshape(-1, dim_thw)  # 重塑为 [batch_size, dim_thw]

    # 处理正弦频率数据
    dim_thw = freqs_sin.shape[-1]  # 获取频率数据的最后一个维度
    freqs_sin = freqs_sin.reshape(temporal_size, h, w, dim_thw)  # 重塑为 [temporal_size, height, width, dim_thw]
    freqs_sin = torch.chunk(freqs_sin, world_size, dim=split_dim - 1)[cur_rank]  # 切分频率数据
    freqs_sin = freqs_sin.reshape(-1, dim_thw)  # 重塑为 [batch_size, dim_thw]

    return latent_model_input, freqs_cos, freqs_sin, split_dim  # 返回处理后的数据


def post_process(output, split_dim):
    """对输出进行后处理，收集所有进程的输出并合并。

    参数:
        output (torch.Tensor): 当前进程的输出，形状为 [batch_size, ...]
        split_dim (int): 切分维度，用于合并输出

    返回:
        torch.Tensor: 合并后的输出，形状为 [world_size * batch_size, ...]
    """
    # 获取当前进程的世界大小
    world_size = dist.get_world_size()

    # 创建一个列表，用于存储所有进程的输出
    gathered_outputs = [torch.empty_like(output) for _ in range(world_size)]

    # 收集所有进程的输出
    dist.all_gather(gathered_outputs, output)

    # 在指定的维度上合并所有进程的输出
    combined_output = torch.cat(gathered_outputs, dim=split_dim)

    return combined_output  # 返回合并后的输出
