import os
import torch
from lightx2v.text2v.models.networks.hunyuan.weights.pre_weights import HunyuanPreWeights
from lightx2v.text2v.models.networks.hunyuan.weights.post_weights import HunyuanPostWeights
from lightx2v.text2v.models.networks.hunyuan.weights.transformer_weights import (
    HunyuanTransformerWeights,
)
from lightx2v.text2v.models.networks.hunyuan.infer.pre_infer import HunyuanPreInfer
from lightx2v.text2v.models.networks.hunyuan.infer.post_infer import HunyuanPostInfer
from lightx2v.text2v.models.networks.hunyuan.infer.transformer_infer import HunyuanTransformerInfer
from lightx2v.text2v.models.networks.hunyuan.infer.feature_caching.transformer_infer import (
    HunyuanTransformerInferFeatureCaching,
)

# from lightx2v.core.distributed.partial_heads_attn.wrap import parallelize_hunyuan
from lightx2v.attentions.distributed.ulysses.wrap import parallelize_hunyuan


class HunyuanModel:
    pre_weight_class = HunyuanPreWeights
    post_weight_class = HunyuanPostWeights
    transformer_weight_class = HunyuanTransformerWeights

    def __init__(self, model_path, config):
        self.model_path = model_path
        self.config = config
        self._init_infer_class()
        self._init_weights()
        self._init_infer()

        if self.config["parallel_attn"]:
            parallelize_hunyuan(self)

        if self.config["cpu_offload"]:
            self.to_cpu()

    def _init_infer_class(self):
        self.pre_infer_class = HunyuanPreInfer
        self.post_infer_class = HunyuanPostInfer
        if self.config["feature_caching"] == "NoCaching":
            self.transformer_infer_class = HunyuanTransformerInfer
        elif self.config["feature_caching"] == "TaylorSeer":
            self.transformer_infer_class = HunyuanTransformerInferFeatureCaching
        else:
            raise NotImplementedError(f"Unsupported feature_caching type: {self.config['feature_caching']}")

    def _load_ckpt(self):
        ckpt_path = os.path.join(self.model_path, "hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt")
        weight_dict = torch.load(ckpt_path, map_location="cuda", weights_only=True)["module"]
        return weight_dict

    def _init_weights(self):
        weight_dict = self._load_ckpt()
        # init weights
        self.pre_weight = self.pre_weight_class(self.config)
        self.post_weight = self.post_weight_class(self.config)
        self.transformer_weights = self.transformer_weight_class(self.config)
        # load weights
        self.pre_weight.load_weights(weight_dict)
        self.post_weight.load_weights(weight_dict)
        self.transformer_weights.load_weights(weight_dict)

    def _init_infer(self):
        self.pre_infer = self.pre_infer_class()
        self.post_infer = self.post_infer_class()
        self.transformer_infer = self.transformer_infer_class(self.config)

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
        self.transformer_infer.set_scheduler(scheduler)

    def to_cpu(self):
        self.pre_weight.to_cpu()
        self.post_weight.to_cpu()
        self.transformer_weights.to_cpu()

    def to_cuda(self):
        self.pre_weight.to_cuda()
        self.post_weight.to_cuda()
        self.transformer_weights.to_cuda()

    @torch.no_grad()
    def infer(self, text_encoder_output, image_encoder_output, args):
        pre_infer_out = self.pre_infer.infer(
            self.pre_weight,
            self.scheduler.latents,
            self.scheduler.timesteps[self.scheduler.step_index],
            text_encoder_output["text_encoder_1_text_states"],
            text_encoder_output["text_encoder_1_attention_mask"],
            text_encoder_output["text_encoder_2_text_states"],
            self.scheduler.freqs_cos,
            self.scheduler.freqs_sin,
            self.scheduler.guidance,
        )
        img, vec = self.transformer_infer.infer(self.transformer_weights, *pre_infer_out)
        self.scheduler.noise_pred = self.post_infer.infer(self.post_weight, img, vec, self.scheduler.latents.shape)
