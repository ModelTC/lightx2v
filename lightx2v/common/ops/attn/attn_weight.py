import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER
import torch.nn.functional as F

try:
    from spas_sage_attn.autotune import SparseAttentionMeansim
except ImportError:
    print("SparseAttentionMeansim not found, please install sparge first")
    SparseAttentionMeansim = None

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except ImportError:
    print("flash_attn_varlen_func not found, please install flash_attn2 first")
    flash_attn_varlen_func = None

try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3
except ImportError:
    print("flash_attn_varlen_func_v3 not found, please install flash_attn3 first")
    flash_attn_varlen_func_v3 = None

if torch.cuda.get_device_capability(0) == (8, 9):
    try:
        from sageattention import sageattn_qk_int8_pv_fp16_triton as sageattn
    except ImportError:
        print("sageattn not found, please install sageattention first")
        sageattn = None, None
else:
    try:
        from sageattention import sageattn
    except ImportError:
        print("sageattn not found, please install sageattention first")
        sageattn = None


class AttnWeightTemplate(metaclass=ABCMeta):
    def __init__(self, weight_name):
        self.weight_name = weight_name
        self.config = {}

    def load(self, weight_dict):
        pass

    @abstractmethod
    def apply(self, input_tensor):
        pass

    def set_config(self, config=None):
        if config is not None:
            self.config = config

    def to_cpu(self, non_blocking=False):
        pass

    def to_cuda(self, non_blocking=False):
        pass

    def state_dict(self, destination=None):
        if destination is None:
            destination = {}
        return destination


@ATTN_WEIGHT_REGISTER("flash_attn2")
class FlashAttn2Weight(AttnWeightTemplate):
    def __init__(self):
        self.config = {}

    def apply(self, q, k, v, cu_seqlens_q=None, cu_seqlens_kv=None, max_seqlen_q=None, max_seqlen_kv=None, model_cls=None):
        x = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        ).reshape(max_seqlen_q, -1)
        return x


@ATTN_WEIGHT_REGISTER("flash_attn3")
class FlashAttn3Weight(AttnWeightTemplate):
    def __init__(self):
        self.config = {}

    def apply(self, q, k, v, cu_seqlens_q=None, cu_seqlens_kv=None, max_seqlen_q=None, max_seqlen_kv=None, model_cls=None):
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


@ATTN_WEIGHT_REGISTER("sage_attn2")
class SageAttn2Weight(AttnWeightTemplate):
    def __init__(self):
        self.config = {}

    def apply(self, q, k, v, cu_seqlens_q=None, cu_seqlens_kv=None, max_seqlen_q=None, max_seqlen_kv=None, model_cls=None):
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
        elif model_cls in ["wan2.1", "wan2.1_causvid", "wan2.1_df"]:
            x = sageattn(
                q.unsqueeze(0),
                k.unsqueeze(0),
                v.unsqueeze(0),
                tensor_layout="NHD",
            )
            x = x.view(max_seqlen_q, -1)
        return x


@ATTN_WEIGHT_REGISTER("torch_sdpa")
class TorchSDPAWeight(AttnWeightTemplate):
    def __init__(self):
        self.config = {}

    def apply(self, q, k, v, drop_rate=0, attn_mask=None, causal=False):
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal)
        x = x.transpose(1, 2)
        b, s, a, d = x.shape
        out = x.reshape(b, s, -1)
        return out


@ATTN_WEIGHT_REGISTER("Sparge")
class SpargeAttnWeight(AttnWeightTemplate):
    def __init__(self, weight_name, verbose=False, l1=0.07, pv_l1=0.08, tune_pv=True, inner_attn_type="flash_attn3"):
        self.verbose = (verbose,)
        self.l1 = (l1,)
        self.pv_l1 = (pv_l1,)
        self.tune_pv = (tune_pv,)
        self.inner_attn_type = inner_attn_type
        self.inner_cls = SparseAttentionMeansim(l1=l1, pv_l1=pv_l1, tune_pv=tune_pv)
        super().__init__(weight_name)

    def load(self, weight_dict):
        # match all key with prefix weight_name
        for key in weight_dict.keys():
            if key.startswith(self.weight_name):
                sub_name = key.split(".")[-1]
                setattr(self.inner_cls, sub_name, nn.Parameter(weight_dict[key], requires_grad=False))

    def apply(self, q, k, v, cu_seqlens_q=None, cu_seqlens_kv=None, max_seqlen_q=None, max_seqlen_kv=None, model_cls=None):
        if len(q.shape) == 3:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)

        x = self.inner_cls(q, k, v, tensor_layout="NHD")
        x = x.flatten(2)
        x = x.squeeze(0)

        return x
