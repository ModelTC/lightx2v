import os
import sys
import torch
import glob
import json
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


class WanModel:
    pre_weight_class = WanPreWeights
    post_weight_class = WanPostWeights
    transformer_weight_class = WanTransformerWeights

    def __init__(self, model_path, config, device):
        self.model_path = model_path
        self.config = config

        self.dit_quantized = self.config.mm_config.get("mm_type", "Default") != "Default"
        self.dit_quantized_ckpt = self.config.get("dit_quantized_ckpt", None)
        self.weight_auto_quant = self.config.mm_config.get("weight_auto_quant", False)
        if self.dit_quantized:
            assert self.weight_auto_quant or self.dit_quantized_ckpt is not None

        self.device = device
        self._init_infer_class()
        self._init_weights()
        self._init_infer()
        self.current_lora = None

        if config["parallel_attn_type"]:
            if config["parallel_attn_type"] == "ulysses":
                ulysses_dist_wrap.parallelize_wan(self)
            elif config["parallel_attn_type"] == "ring":
                ring_dist_wrap.parallelize_wan(self)
            elif config["parallel_attn_type"] == "pipefusion":
                pass
            else:
                raise Exception(f"Unsuppotred parallel_attn_type")

    def _init_infer_class(self):
        self.pre_infer_class = WanPreInfer
        self.post_infer_class = WanPostInfer
        if self.config["feature_caching"] == "NoCaching":
            self.transformer_infer_class = WanTransformerInfer
        elif self.config["feature_caching"] == "Tea":
            self.transformer_infer_class = WanTransformerInferTeaCaching
        else:
            raise NotImplementedError(f"Unsupported feature_caching type: {self.config['feature_caching']}")

    def _load_safetensor_to_dict(self, file_path):
        use_bfloat16 = self.config.get("use_bfloat16", True)
        with safe_open(file_path, framework="pt") as f:
            if use_bfloat16:
                tensor_dict = {key: f.get_tensor(key).pin_memory().to(torch.bfloat16).to(self.device) for key in f.keys()}
            else:
                tensor_dict = {key: f.get_tensor(key).pin_memory().to(self.device) for key in f.keys()}
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

    def _load_quant_ckpt(self):
        ckpt_path = self.config.dit_quantized_ckpt
        logger.info(f"Loading quant dit model from {ckpt_path}")

        if ckpt_path.endswith(".pth"):
            logger.info(f"Loading {ckpt_path} as PyTorch model.")
            weight_dict = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        else:
            index_files = [f for f in os.listdir(ckpt_path) if f.endswith(".index.json")]
            if not index_files:
                raise FileNotFoundError(f"No .pth file or *.index.json found in {ckpt_path}")

            index_path = os.path.join(ckpt_path, index_files[0])
            logger.info(f" Using safetensors index: {index_path}")

            with open(index_path, "r") as f:
                index_data = json.load(f)

            weight_dict = {}
            for filename in set(index_data["weight_map"].values()):
                safetensor_path = os.path.join(ckpt_path, filename)
                with safe_open(safetensor_path, framework="pt", device=str(self.device)) as f:
                    logger.info(f"Loading weights from {safetensor_path}")
                    for k in f.keys():
                        weight_dict[k] = f.get_tensor(k).pin_memory()
                        if weight_dict[k].dtype == torch.float:
                            weight_dict[k] = weight_dict[k].pin_memory().to(torch.bfloat16)

        return weight_dict

    def _load_quant_split_ckpt(self):
        lazy_load_model_path = self.config.dit_quantized_ckpt
        logger.info(f"Loading splited quant model from {lazy_load_model_path}")
        pre_post_weight_dict, transformer_weight_dict = {}, {}

        safetensor_path = os.path.join(lazy_load_model_path, "non_block.safetensors")
        with safe_open(safetensor_path, framework="pt", device=str(self.device)) as f:
            for k in f.keys():
                pre_post_weight_dict[k] = f.get_tensor(k).pin_memory()
                if pre_post_weight_dict[k].dtype == torch.float:
                    pre_post_weight_dict[k] = pre_post_weight_dict[k].pin_memory().to(torch.bfloat16)

        safetensors_pattern = os.path.join(lazy_load_model_path, "block_*.safetensors")
        safetensors_files = glob.glob(safetensors_pattern)
        if not safetensors_files:
            raise FileNotFoundError(f"No .safetensors files found in directory: {lazy_load_model_path}")

        for file_path in safetensors_files:
            with safe_open(file_path, framework="pt") as f:
                for k in f.keys():
                    if "modulation" in k:
                        transformer_weight_dict[k] = f.get_tensor(k).pin_memory()
                        if transformer_weight_dict[k].dtype == torch.float:
                            transformer_weight_dict[k] = transformer_weight_dict[k].pin_memory().to(torch.bfloat16)

        return pre_post_weight_dict, transformer_weight_dict

    def _init_weights(self, weight_dict=None):
        if weight_dict is None:
            if not self.dit_quantized or self.weight_auto_quant:
                self.original_weight_dict = self._load_ckpt()
            else:
                if not self.config.get("lazy_load", False):
                    self.original_weight_dict = self._load_quant_ckpt()
                else:
                    (
                        self.original_weight_dict,
                        self.transformer_weight_dict,
                    ) = self._load_quant_split_ckpt()
        else:
            self.original_weight_dict = weight_dict

        # init weights
        self.pre_weight = self.pre_weight_class(self.config)
        self.post_weight = self.post_weight_class(self.config)
        self.transformer_weights = self.transformer_weight_class(self.config)
        # load weights
        self.pre_weight.load(self.original_weight_dict)
        self.post_weight.load(self.original_weight_dict)
        if hasattr(self, "transformer_weight_dict"):
            self.transformer_weights.load(self.transformer_weight_dict)
        else:
            self.transformer_weights.load(self.original_weight_dict)

    def _init_infer(self):
        self.pre_infer = self.pre_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)
        self.transformer_infer = self.transformer_infer_class(self.config)

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
        self.pre_infer.set_scheduler(scheduler)
        self.post_infer.set_scheduler(scheduler)
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
    def infer(self, inputs):
        if self.config["cpu_offload"]:
            self.pre_weight.to_cuda()
            self.post_weight.to_cuda()

        embed, grid_sizes, pre_infer_out = self.pre_infer.infer(self.pre_weight, inputs, positive=True)
        x = self.transformer_infer.infer(self.transformer_weights, grid_sizes, embed, *pre_infer_out)
        noise_pred_cond = self.post_infer.infer(self.post_weight, x, embed, grid_sizes)[0]

        if self.config["feature_caching"] == "Tea":
            self.scheduler.cnt += 1
            if self.scheduler.cnt >= self.scheduler.num_steps:
                self.scheduler.cnt = 0
        self.scheduler.noise_pred = noise_pred_cond

        if self.config["enable_cfg"]:
            embed, grid_sizes, pre_infer_out = self.pre_infer.infer(self.pre_weight, inputs, positive=False)
            x = self.transformer_infer.infer(self.transformer_weights, grid_sizes, embed, *pre_infer_out)
            noise_pred_uncond = self.post_infer.infer(self.post_weight, x, embed, grid_sizes)[0]

            if self.config["feature_caching"] == "Tea":
                self.scheduler.cnt += 1
                if self.scheduler.cnt >= self.scheduler.num_steps:
                    self.scheduler.cnt = 0

            self.scheduler.noise_pred = noise_pred_uncond + self.config.sample_guide_scale * (noise_pred_cond - noise_pred_uncond)

            if self.config["cpu_offload"]:
                self.pre_weight.to_cpu()
                self.post_weight.to_cpu()
