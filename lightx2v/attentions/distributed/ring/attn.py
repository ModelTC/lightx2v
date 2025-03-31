import torch
import torch.distributed as dist
from lightx2v.attentions import attention


def ring_attn(q, k, v, img_qkv_len, cu_seqlens_qkv, attention_type="flash_attn2"):
    """
    执行 Ulysses 注意力机制，结合图像和文本的查询、键和值。

    参数:
        q (torch.Tensor): 查询张量，形状为 [shard_seqlen, heads, hidden_dims]
        k (torch.Tensor): 键张量，形状为 [shard_seqlen, heads, hidden_dims]
        v (torch.Tensor): 值张量，形状为 [shard_seqlen, heads, hidden_dims]
        img_qkv_len (int): 图像查询、键和值的长度
        cu_seqlens_qkv (torch.Tensor): 累积序列长度，包含文本和图像的长度信息
        attention_type (str): 注意力类型，默认为 "flash_attn2"

    返回:
        torch.Tensor: 计算得到的注意力结果
    """
    # 获取当前进程的排名和全局进程数
    cur_rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 获取序列长度和文本相关的长度
    seq_len = q.shape[0]
    txt_qkv_len = cu_seqlens_qkv[1] - img_qkv_len  # 文本查询、键和值的长度
    txt_mask_len = cu_seqlens_qkv[2] - img_qkv_len  # 文本掩码长度

    # 获取查询张量的头数和隐藏维度
    _, heads, hidden_dims = q.shape
    shard_heads = heads // world_size  # 每个进程处理的头数
    shard_seqlen = img_qkv_len  # 每个进程处理的序列长度

    # 分割图像和文本的查询、键和值
    img_q, img_k, img_v = (
        q[:img_qkv_len, :, :].contiguous(),
        k[:img_qkv_len, :, :].contiguous(),
        v[:img_qkv_len, :, :].contiguous(),
    )
    txt_q, txt_k, txt_v = (
        q[img_qkv_len:, :, :].contiguous(),
        k[img_qkv_len:, :, :].contiguous(),
        v[img_qkv_len:, :, :].contiguous(),
    )

    gathered_img_k = [torch.empty_like(img_k) for _ in range(world_size)]
    gathered_img_v = [torch.empty_like(img_v) for _ in range(world_size)]

    dist.all_gather(gathered_img_k, img_k)
    dist.all_gather(gathered_img_v, img_v)
    torch.cuda.synchronize()

    q = q
    k = torch.cat(gathered_img_k + [txt_k], dim=0)
    v = torch.cat(gathered_img_v + [txt_v], dim=0)

    # 初始化累积序列长度张量
    cu_seqlens_q = torch.zeros([3], dtype=torch.int32, device="cuda")
    s = txt_qkv_len + img_q.shape[0]  # 计算文本和图像的总长度
    s1 = s  # 当前样本的结束位置
    s2 = txt_mask_len + img_q.shape[0]  # 文本掩码的结束位置
    cu_seqlens_q[1] = s1  # 设置累积序列长度
    cu_seqlens_q[2] = s2  # 设置累积序列长度
    max_seqlen_q = img_q.shape[0] + txt_q.shape[0]  # 最大序列长度

    # 初始化累积序列长度张量
    cu_seqlens_kv = torch.zeros([3], dtype=torch.int32, device="cuda")
    s = txt_qkv_len + img_k.shape[0] * world_size  # 计算文本和图像的总长度
    s1 = s  # 当前样本的结束位置
    s2 = txt_mask_len + img_k.shape[0] * world_size  # 文本掩码的结束位置
    cu_seqlens_kv[1] = s1  # 设置累积序列长度
    cu_seqlens_kv[2] = s2  # 设置累积序列长度
    max_seqlen_kv = img_k.shape[0] * world_size + txt_q.shape[0]  # 最大序列长度

    attn = attention(
        attention_type=attention_type,
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
    )

    return attn
