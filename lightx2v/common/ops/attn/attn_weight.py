import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER
from lightx2v.attentions import attention
from spas_sage_attn.autotune import SparseAttentionMeansim


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
        self.weight = self.weight.to("cpu", non_blocking=non_blocking)

    def to_cuda(self, non_blocking=False):
        self.weight = self.weight.cuda(non_blocking=non_blocking)


@ATTN_WEIGHT_REGISTER("Default")
class DefaultAttnWeightTemplate(AttnWeightTemplate):
    def __init__(self, attn_type):
        self.attn_type = attn_type
        self.config = {}

    def load(self, weight_dict):
        pass

    def apply(self, q, k, v, cu_seqlens_q=None, cu_seqlens_kv=None, max_seqlen_q=None, max_seqlen_kv=None, model_cls=None):
        return attention(self.attn_type, q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv, max_seqlen_q=max_seqlen_q, max_seqlen_kv=max_seqlen_kv, model_cls=model_cls)

    def set_config(self, config=None):
        if config is not None:
            self.config = config


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
