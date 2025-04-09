import torch

if torch.cuda.get_device_capability(0) == (8, 9):
    try:
        from sageattention import sageattn_qk_int8_pv_fp16_triton as sageattn
    except ImportError:
        sageattn = None, None
else:
    try:
        from sageattention import sageattn
    except ImportError:
        sageattn = None


def sage_attn2(q, k, v, cu_seqlens_q=None, cu_seqlens_kv=None, max_seqlen_q=None, max_seqlen_kv=None, model_cls="hunyuan"):
    q, k, v = (
        q.transpose(1, 0).contiguous(),
        k.transpose(1, 0).contiguous(),
        v.transpose(1, 0).contiguous(),
    )

    if model_cls == "hunyuan":
        x1 = sageattn(
            q[:, : cu_seqlens_q[1], :].unsqueeze(0),
            k[:, : cu_seqlens_q[1], :].unsqueeze(0),
            v[:, : cu_seqlens_kv[1], :].unsqueeze(0),
        )
        x2 = sageattn(
            q[:, cu_seqlens_q[1] :, :].unsqueeze(0),
            k[:, cu_seqlens_kv[1] :, :].unsqueeze(0),
            v[:, cu_seqlens_kv[1] :, :].unsqueeze(0),
        )
        x = torch.cat((x1, x2), dim=-2).transpose(2, 1).contiguous()
        x = x.view(max_seqlen_q, -1)
    elif model_cls == "wan2.1":
        x = (
            sageattn(
                q[:, : cu_seqlens_q[1], :].unsqueeze(0),
                k[:, : cu_seqlens_q[1], :].unsqueeze(0),
                v[:, : cu_seqlens_kv[1], :].unsqueeze(0),
            )
            .transpose(2, 1)
            .contiguous()
        )
        x = x.view(max_seqlen_q, -1)
    return x
