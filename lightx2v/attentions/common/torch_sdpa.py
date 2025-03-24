import torch
import torch.nn.functional as F


def torch_sdpa(
    q,
    k,
    v,
    drop_rate=0,
    attn_mask=None,
    causal=False,
):
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    if attn_mask is not None and attn_mask.dtype != torch.bool:
        attn_mask = attn_mask.to(q.dtype)
    x = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal
    )
    x = x.transpose(1, 2)
    b, s, a, d = x.shape
    out = x.reshape(b, s, -1)
    return out
