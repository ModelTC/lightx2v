try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3
except ImportError:
    flash_attn_varlen_func_v3 = None


def flash_attn3(
    q, k, v, cu_seqlens_q=None, cu_seqlens_kv=None, max_seqlen_q=None, max_seqlen_kv=None
):
    x = flash_attn_varlen_func_v3(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
    )[0].reshape(max_seqlen_q, -1)
    return x
