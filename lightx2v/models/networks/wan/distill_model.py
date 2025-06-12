import os
import sys
import torch
import glob
import json
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.networks.wan.weights.pre_weights import WanPreWeights
from lightx2v.models.networks.wan.weights.post_weights import WanPostWeights
from lightx2v.models.networks.wan.weights.transformer_weights import (
    WanTransformerWeights,
)
from lightx2v.models.networks.wan.infer.pre_infer import WanPreInfer
from lightx2v.models.networks.wan.infer.post_infer import WanPostInfer
from lightx2v.models.networks.wan.infer.transformer_infer import (
    WanTransformerInfer,
)
from lightx2v.models.networks.wan.infer.feature_caching.transformer_infer import (
    WanTransformerInferTeaCaching,
)
from safetensors import safe_open
import lightx2v.attentions.distributed.ulysses.wrap as ulysses_dist_wrap
import lightx2v.attentions.distributed.ring.wrap as ring_dist_wrap
from lightx2v.utils.envs import *
from loguru import logger


class WanDistillModel(WanModel):
    pre_weight_class = WanPreWeights
    post_weight_class = WanPostWeights
    transformer_weight_class = WanTransformerWeights

    def __init__(self, model_path, config, device):
        super().__init__(model_path, config, device)

    def _load_ckpt(self):
        use_bfloat16 = self.config.get("use_bfloat16", True)
        ckpt_path = os.path.join(self.model_path, "distill_model.pt")
        if not os.path.exists(ckpt_path):
            # 文件不存在，调用父类的 _load_ckpt 方法
            return super()._load_ckpt()

        weight_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        dtype = torch.bfloat16 if use_bfloat16 else None
        for key, value in weight_dict.items():
            weight_dict[key] = value.to(device=self.device, dtype=dtype)

        return weight_dict
