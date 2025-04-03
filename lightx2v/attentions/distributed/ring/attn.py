import torch
import torch.distributed as dist
import torch.nn.functional as F

# from lightx2v.attentions import attention
from lightx2v.attentions.distributed.comm.ring_comm import RingComm
import flash_attn
from flash_attn.flash_attn_interface import _flash_attn_forward
from typing import Optional, Tuple


# RING_COMM = None


# def init_ring_comm():
#     global RING_COMM
#     RING_COMM = RingComm()


@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    # new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    # torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
    # For additional context and discussion, please refer to:
    # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)

    return out, lse


def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(slice_out, slice_lse, block_out, block_lse)
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse


def ring_attn_sub(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None, return_softmax=False):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if flash_attn.__version__ < "2.6.3":
        block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax,
        )
    else:
        block_out, block_lse, _, _ = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax,
        )
    return block_out, block_lse


def ring_attn(q, k, v, img_qkv_len, cu_seqlens_qkv, attention_type="flash_attn2"):
    """
    执行 Ring 注意力机制，结合图像和文本的查询、键和值。

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
    # print('ring attn')
    # 获取当前进程的排名和全局进程数
    cur_rank = dist.get_rank()
    world_size = dist.get_world_size()

    # if cur_rank == 0:
    #     import pdb; pdb.set_trace()
    # import time; time.sleep(9999)

    if len(cu_seqlens_qkv) == 3:
        txt_qkv_len = cu_seqlens_qkv[1] - img_qkv_len  # 文本查询、键和值的长度
        txt_mask_len = cu_seqlens_qkv[2] - img_qkv_len  # 文本掩码长度
    elif len(cu_seqlens_qkv) == 2:
        txt_qkv_len = cu_seqlens_qkv[1] - img_qkv_len  # 文本查询、键和值的长度
        txt_mask_len = 0

    # if RING_COMM is None:
    #     init_ring_comm()

    RING_COMM = RingComm()

    # if len(cu_seqlens_qkv) == 3:
    #     txt_qkv_len = cu_seqlens_qkv[1] - img_qkv_len  # 文本查询、键和值的长度
    #     txt_mask_len = cu_seqlens_qkv[2] - img_qkv_len  # 文本掩码长度
    # elif len(cu_seqlens_qkv) == 2:
    #     txt_qkv_len = cu_seqlens_qkv[1] - img_qkv_len  # 文本查询、键和值的长度
    #     txt_mask_len = None
    q = q.unsqueeze(0)
    k = k.unsqueeze(0)
    v = v.unsqueeze(0)

    img_q, img_k, img_v = q[:, :img_qkv_len, :, :].contiguous(), k[:, :img_qkv_len, :, :].contiguous(), v[:, :img_qkv_len, :, :].contiguous()
    txt_q, txt_k, txt_v = (
        q[:, img_qkv_len : img_qkv_len + txt_qkv_len, :, :].contiguous(),
        k[:, img_qkv_len : img_qkv_len + txt_qkv_len, :, :].contiguous(),
        v[:, img_qkv_len : img_qkv_len + txt_qkv_len, :, :].contiguous(),
    )

    out, lse, next_k, next_v = None, None, None, None

    if len(cu_seqlens_qkv) == 3:
        q = torch.cat((img_q, txt_q), dim=1)
    k = img_k
    v = img_v

    for step in range(world_size):
        if step + 1 != world_size:
            next_k = RING_COMM.send_recv(k)
            next_v = RING_COMM.send_recv(v)
            RING_COMM.commit()

        if step + 1 == world_size:
            k = torch.cat((k, txt_k), dim=1)
            v = torch.cat((v, txt_v), dim=1)

        block_out, block_lse = ring_attn_sub(q, k, v)

        out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        if step + 1 != world_size:
            RING_COMM.wait()
            k = next_k
            v = next_v

    attn1 = out.to(torch.bfloat16).squeeze(0).reshape(img_qkv_len + txt_qkv_len, -1)

    if txt_mask_len > 0:
        attn2, *_ = _flash_attn_forward(
            q[:, -(txt_mask_len - txt_qkv_len) :, :, :].contiguous(),
            k[:, -(txt_mask_len - txt_qkv_len) :, :, :].contiguous(),
            v[:, -(txt_mask_len - txt_qkv_len) :, :, :].contiguous(),
            dropout_p=0.0,
            softmax_scale=q.shape[-1] ** (-0.5),
            causal=False,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            return_softmax=False,
        )

        attn2 = attn2.to(torch.bfloat16).squeeze(0).reshape((txt_mask_len - txt_qkv_len), -1)
        attn1 = torch.cat([attn1, attn2], dim=0)

    return attn1
