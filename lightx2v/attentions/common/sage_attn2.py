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
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    if model_cls == "hunyuan":
        x1 = sageattn(
            q[: cu_seqlens_q[1]].unsqueeze(0),
            k[: cu_seqlens_kv[1]].unsqueeze(0),
            v[: cu_seqlens_kv[1]].unsqueeze(0),
            tensor_layout="NHD",
        )
        x2 = sageattn(
            q[cu_seqlens_q[1] :].unsqueeze(0),
            k[cu_seqlens_kv[1] :].unsqueeze(0),
            v[cu_seqlens_kv[1] :].unsqueeze(0),
            tensor_layout="NHD",
        )
        x = torch.cat((x1, x2), dim=1)
        x = x.view(max_seqlen_q, -1)
    elif model_cls in ["wan2.1", "wan2.1_distill", "wan2.1_causvid", "wan2.1_df"]:
        x = sageattn(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            tensor_layout="NHD",
        )
        x = x.view(max_seqlen_q, -1)
    return x
