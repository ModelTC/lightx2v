import os
import torch
import time
import glob
from lightx2v.text2v.models.networks.wan.weights.pre_weights import WanPreWeights
from lightx2v.text2v.models.networks.wan.weights.post_weights import WanPostWeights
from lightx2v.text2v.models.networks.wan.weights.transformer_weights import (
    WanTransformerWeights,
)
from lightx2v.text2v.models.networks.wan.infer.pre_infer import WanPreInfer
from lightx2v.text2v.models.networks.wan.infer.post_infer import WanPostInfer
from lightx2v.text2v.models.networks.wan.infer.transformer_infer import (
    WanTransformerInfer,
)
from lightx2v.text2v.models.networks.wan.infer.feature_caching.transformer_infer import WanTransformerInferFeatureCaching
from safetensors import safe_open
import lightx2v.attentions.distributed.ulysses.wrap as ulysses_dist_wrap
import lightx2v.attentions.distributed.ring.wrap as ring_dist_wrap


class WanModel:
    pre_weight_class = WanPreWeights
    post_weight_class = WanPostWeights
    transformer_weight_class = WanTransformerWeights

    def __init__(self, model_path, config):
        self.model_path = model_path
        self.config = config
        self._init_infer_class()
        self._init_weights()
        self._init_infer()

        if config["parallel_attn_type"]:
            if config["parallel_attn_type"] == "ulysses":
                ulysses_dist_wrap.parallelize_wan(self)
            elif config["parallel_attn_type"] == "ring":
                ring_dist_wrap.parallelize_wan(self)
            else:
                raise Exception(f"Unsuppotred parallel_attn_type")

        if self.config["cpu_offload"]:
            self.to_cpu()

    def _init_infer_class(self):
        self.pre_infer_class = WanPreInfer
        self.post_infer_class = WanPostInfer
        if self.config["feature_caching"] == "NoCaching":
            self.transformer_infer_class = WanTransformerInfer
        elif self.config["feature_caching"] == "Tea":
            self.transformer_infer_class = WanTransformerInferFeatureCaching
        else:
            raise NotImplementedError(f"Unsupported feature_caching type: {self.config['feature_caching']}")

    def _load_safetensor_to_dict(self, file_path):
        with safe_open(file_path, framework="pt") as f:
            tensor_dict = {key: f.get_tensor(key).to(torch.bfloat16).cuda() for key in f.keys()}
        return tensor_dict

    def _load_ckpt(self):
        safetensors_pattern = os.path.join(self.model_path, "*.safetensors")
        safetensors_files = glob.glob(safetensors_pattern)

        if not safetensors_files:
            raise FileNotFoundError(f"No .safetensors files found in directory: {self.model_path}")
        weight_dict = {}
        for file_path in safetensors_files:
            file_weights = self._load_safetensor_to_dict(file_path)
            weight_dict.update(file_weights)
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
        self.pre_infer = self.pre_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)
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
    def infer(self, text_encoders_output, image_encoder_output, args):
        timestep = torch.stack([self.scheduler.timesteps[self.scheduler.step_index]])

        embed, grid_sizes, pre_infer_out = self.pre_infer.infer(
            self.pre_weight,
            [self.scheduler.latents],
            timestep,
            text_encoders_output["context"],
            self.scheduler.seq_len,
            image_encoder_output["clip_encoder_out"],
            [image_encoder_output["vae_encode_out"]],
        )
        x = self.transformer_infer.infer(self.transformer_weights, grid_sizes, embed, *pre_infer_out)
        noise_pred_cond = self.post_infer.infer(self.post_weight, x, embed, grid_sizes)[0]

        if self.config["feature_caching"] == "Tea":
            self.scheduler.cnt += 1
            if self.scheduler.cnt >= self.scheduler.num_steps:
                self.scheduler.cnt = 0

        embed, grid_sizes, pre_infer_out = self.pre_infer.infer(
            self.pre_weight,
            [self.scheduler.latents],
            timestep,
            text_encoders_output["context_null"],
            self.scheduler.seq_len,
            image_encoder_output["clip_encoder_out"],
            [image_encoder_output["vae_encode_out"]],
        )
        x = self.transformer_infer.infer(self.transformer_weights, grid_sizes, embed, *pre_infer_out)
        noise_pred_uncond = self.post_infer.infer(self.post_weight, x, embed, grid_sizes)[0]

        if self.config["feature_caching"] == "Tea":
            self.scheduler.cnt += 1
            if self.scheduler.cnt >= self.scheduler.num_steps:
                self.scheduler.cnt = 0

        self.scheduler.noise_pred = noise_pred_uncond + args.sample_guide_scale * (noise_pred_cond - noise_pred_uncond)
