import torch
import torch.distributed as dist
from lightx2v.attentions import attention
from lightx2v.utils.utils import seed_all


seed_all(42)


def prepare_tensors():
    cur_rank = dist.get_rank()  # 获取当前进程的 rank
    torch.cuda.set_device(cur_rank)  # 设置当前进程的 CUDA 设备
    q = torch.randn(32656, 24, 128, dtype=torch.bfloat16).cuda()
    k = torch.randn(32656, 24, 128, dtype=torch.bfloat16).cuda()
    v = torch.randn(32656, 24, 128, dtype=torch.bfloat16).cuda()

    cu_seqlens_qkv = torch.tensor([0, 32411, 32656], dtype=torch.int32).cuda()
    max_seqlen_qkv = 32656
    return q, k, v, cu_seqlens_qkv, max_seqlen_qkv


def test_part_head():
    q, k, v, cu_seqlens_qkv, max_seqlen_qkv = prepare_tensors()

    # 先计算完整的结果作为参考
    single_gpu_output = attention(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_qkv,
        cu_seqlens_kv=cu_seqlens_qkv,
        max_seqlen_q=max_seqlen_qkv,
        max_seqlen_kv=max_seqlen_qkv,
    )

    num_heads = q.shape[-2]
    cur_rank = dist.get_rank()
    world_size = dist.get_world_size()
    num_chunk_heads = int(num_heads / dist.get_world_size())

    if cur_rank == world_size - 1:
        q = q[:, num_chunk_heads * cur_rank :, :]
        k = k[:, num_chunk_heads * cur_rank :, :]
        v = v[:, num_chunk_heads * cur_rank :, :]
    else:
        q = q[:, num_chunk_heads * cur_rank : num_chunk_heads * (cur_rank + 1), :]
        k = k[:, num_chunk_heads * cur_rank : num_chunk_heads * (cur_rank + 1), :]
        v = v[:, num_chunk_heads * cur_rank : num_chunk_heads * (cur_rank + 1), :]

    output = attention(
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

    # 验证结果一致性
    if cur_rank == 0:
        # import pdb; pdb.set_trace()
        print(
            "Outputs match:",
            torch.allclose(single_gpu_output, combined_output, rtol=1e-3, atol=1e-3),
        )

    # # 验证结果一致性
    # print("Outputs match:", torch.allclose(single_gpu_output, combined_output, rtol=1e-3, atol=1e-3))


if __name__ == "__main__":
    # 初始化分布式环境
    dist.init_process_group(backend="nccl")
    test_part_head()
