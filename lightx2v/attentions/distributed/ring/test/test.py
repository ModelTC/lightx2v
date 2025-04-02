import torch
from lightx2v.attentions import attention
from lightx2v.attentions.distributed.ring.attn import ring_attn_sub, update_out_and_lse


def attention(q, k, v, cu_seqlens_q, cu_seqlens_k, lq, lk):
    attn_out = attention(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_k,
        max_seqlen_q=lq,
        max_seqlen_kv=lk,
    )
    return attn_out


def ring_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, lq, lk, ring_size):
    out, lse = None, None
    # q = torch.chunk(q, ring_size)
    k = torch.chunk(k, ring_size)
    v = torch.chunk(v, ring_size)
    for i in range(ring_size):
        k_block, v_block = k[i], v[i]
        block_out, block_lse = ring_attn_sub(q, k_block, v_block)
        out, lse = update_out_and_lse(out, lse, block_out, block_lse)
    return out


def test():
    q = torch.randn((32760, 12, 128), dtype=torch.bfloat16, device='cuda')
    k = torch.randn((32760, 12, 128), dtype=torch.bfloat16, device='cuda')
    v = torch.randn((32760, 12, 128), dtype=torch.bfloat16, device='cuda')
    cu_seqlens_q = torch.tensor([0, 32760], dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, 32760], dtype=torch.int32)
    lq = 32760
    lk = 32760

    base_attn = attention(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        lq=lq,
        lk=lk
    )

    ring_attn = ring_attention(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        lq=lq,
        lk=lk,
        ring_size=4
    )

    # 添加断言以确认数值相同
    assert torch.allclose(base_attn, ring_attn), "base_attn 和 ring_attn 的数值不相同！"