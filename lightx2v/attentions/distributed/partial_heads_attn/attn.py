import torch
import torch.distributed as dist
from lightx2v.attentions import attention


def partial_heads_attn(attention_type, q, k, v, cu_seqlens_qkv, max_seqlen_qkv):
    num_heads = q.shape[-2]
    cur_rank = dist.get_rank()
    world_size = dist.get_world_size()
    num_chunk_heads = int(num_heads / dist.get_world_size())

    if cur_rank == world_size-1:
        q = q[:, num_chunk_heads*cur_rank:, :]
        k = k[:, num_chunk_heads*cur_rank:, :]
        v = v[:, num_chunk_heads*cur_rank:, :]
    else:
        q = q[:, num_chunk_heads*cur_rank:num_chunk_heads*(cur_rank+1), :]
        k = k[:, num_chunk_heads*cur_rank:num_chunk_heads*(cur_rank+1), :]
        v = v[:, num_chunk_heads*cur_rank:num_chunk_heads*(cur_rank+1), :]

    output = attention(
        attention_type=attention_type,
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_qkv,
        cu_seqlens_kv=cu_seqlens_qkv,
        max_seqlen_q=max_seqlen_qkv,
        max_seqlen_kv=max_seqlen_qkv,
    )

    gathered_outputs = [torch.empty_like(output) for _ in range(world_size)]
    dist.all_gather(gathered_outputs, output)

    combined_output = torch.cat(gathered_outputs, dim=1)

    return combined_output