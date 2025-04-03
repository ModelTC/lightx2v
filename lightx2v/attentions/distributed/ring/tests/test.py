import torch
import torch.distributed as dist
from lightx2v.attentions import attention
from flash_attn.flash_attn_interface import _flash_attn_varlen_forward
from lightx2v.attentions.distributed.ring.attn import ring_attn_sub, update_out_and_lse
from lightx2v.attentions.distributed.comm.ring_comm import RingComm

RING_COMM = None

def init_ring_comm():
    global RING_COMM
    RING_COMM = RingComm()


def base_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, lq, lk):
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
    q = q.unsqueeze(0)
    k = k.unsqueeze(0)
    v = v.unsqueeze(0)
    
    k = torch.chunk(k, ring_size, dim=1)
    v = torch.chunk(v, ring_size, dim=1)
    
    for i in range(ring_size):
        k_block, v_block = k[i], v[i]
        block_out, block_lse = ring_attn_sub(q, k_block, v_block)
        out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        
    attn_out = out.to(torch.bfloat16).squeeze(0).reshape(lq, -1)
    return attn_out

def ring_attention_dist(q, k, v, cu_seqlens_q, cu_seqlens_k, lq, lk):
    if RING_COMM is None:
        init_ring_comm()
    
    out, lse = None, None
    # q = torch.chunk(q, ring_size)
    cur_rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    out, lse, next_k, next_v = None, None, None, None
    
    q = q.unsqueeze(0)
    k = k.unsqueeze(0)
    v = v.unsqueeze(0)
    
    k = torch.chunk(k, world_size, dim=1)[cur_rank]
    v = torch.chunk(v, world_size, dim=1)[cur_rank]
    
    for step in range(world_size):
        if step + 1 != world_size:
            next_k = RING_COMM.send_recv(k)
            next_v = RING_COMM.send_recv(v)
            RING_COMM.commit()
        block_out, block_lse = ring_attn_sub(q, k, v)
        out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        
        if step + 1 != world_size:
            RING_COMM.wait()
            k = next_k
            v = next_v
        
    attn_out = out.to(torch.bfloat16).squeeze(0).reshape(lq, -1)
    return attn_out


def test():
    q = torch.randn((32760, 12, 128), dtype=torch.bfloat16, device='cuda')
    k = torch.randn((32760, 12, 128), dtype=torch.bfloat16, device='cuda')
    v = torch.randn((32760, 12, 128), dtype=torch.bfloat16, device='cuda')
    cu_seqlens_q = torch.tensor([0, 32760], dtype=torch.int32, device='cuda')
    cu_seqlens_k = torch.tensor([0, 32760], dtype=torch.int32, device='cuda')
    lq = 32760
    lk = 32760

    base_attn = base_attention(
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
    # import pdb; pdb.set_trace()
    # 添加断言以确认数值相同
    assert torch.allclose(base_attn, ring_attn, rtol=1e-3, atol=1e-3), "base_attn 和 ring_attn 的数值不相同！"
    

if __name__ == '__main__':
    # dist.init_process_group(backend="nccl")
    test()