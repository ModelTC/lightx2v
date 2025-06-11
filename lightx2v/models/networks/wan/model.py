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
from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
from lightx2v.models.networks.wan.infer.feature_caching.transformer_infer import WanTransformerInferTeaCaching, WanTransformerInferTaylorCaching, WanTransformerInferAdaCaching
from safetensors import safe_open
import lightx2v.attentions.distributed.ulysses.wrap as ulysses_dist_wrap
import lightx2v.attentions.distributed.ring.wrap as ring_dist_wrap
from lightx2v.utils.envs import *
from lightx2v.utils.memory_profiler import peak_memory_decorator
from loguru import logger


class WanModel:
    pre_weight_class = WanPreWeights
    post_weight_class = WanPostWeights
    transformer_weight_class = WanTransformerWeights

    @peak_memory_decorator
    def __init__(self, model_path, config, device):
        self.model_path = model_path
        self.config = config
        self.device = device
        self._init_infer_class()
        self._init_weights()
        if GET_RUNNING_FLAG() == "save_naive_quant":
            assert self.config.get("quant_model_path") is not None, "quant_model_path is None"
            self.save_weights(self.config.quant_model_path)
            sys.exit(0)

        self._init_infer()
        self.current_lora = None

        if config["parallel_attn_type"]:
            if config["parallel_attn_type"] == "ulysses":
                ulysses_dist_wrap.parallelize_wan(self)
            elif config["parallel_attn_type"] == "ring":
                ring_dist_wrap.parallelize_wan(self)
            else:
                raise Exception(f"Unsuppotred parallel_attn_type")

    def _init_infer_class(self):
        self.pre_infer_class = WanPreInfer
        self.post_infer_class = WanPostInfer
        if self.config["feature_caching"] == "NoCaching":
            self.transformer_infer_class = WanTransformerInfer
        elif self.config["feature_caching"] == "Tea":
            self.transformer_infer_class = WanTransformerInferTeaCaching
        elif self.config["feature_caching"] == "Taylor":
            self.transformer_infer_class = WanTransformerInferTaylorCaching
        elif self.config["feature_caching"] == "Ada":
            self.transformer_infer_class = WanTransformerInferAdaCaching
        else:
            raise NotImplementedError(f"Unsupported feature_caching type: {self.config['feature_caching']}")

    def _load_safetensor_to_dict(self, file_path):
        use_bfloat16 = self.config.get("use_bfloat16", True)
        with safe_open(file_path, framework="pt") as f:
            if use_bfloat16:
                tensor_dict = {key: f.get_tensor(key).to(torch.bfloat16).to(self.device) for key in f.keys()}
            else:
                tensor_dict = {key: f.get_tensor(key).to(self.device) for key in f.keys()}
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
        assert self.config.get("quant_model_path") is not None, "quant_model_path is None"
        ckpt_path = self.config.quant_model_path
        logger.info(f"Loading quant model from {ckpt_path}")

        quant_pth_file = os.path.join(ckpt_path, "quant_weights.pth")

        if os.path.exists(quant_pth_file):
            logger.info("Found quant_weights.pth, loading as PyTorch model.")
            weight_dict = torch.load(quant_pth_file, map_location=self.device, weights_only=True)
        else:
            index_files = [f for f in os.listdir(ckpt_path) if f.endswith(".index.json")]
            if not index_files:
                raise FileNotFoundError(f"No quant_weights.pth or *.index.json found in {ckpt_path}")

            index_path = os.path.join(ckpt_path, index_files[0])
            logger.info(f"quant_weights.pth not found. Using safetensors index: {index_path}")

            with open(index_path, "r") as f:
                index_data = json.load(f)

            weight_dict = {}
            for filename in set(index_data["weight_map"].values()):
                safetensor_path = os.path.join(ckpt_path, filename)
                with safe_open(safetensor_path, framework="pt", device=str(self.device)) as f:
                    logger.info(f"Loading weights from {safetensor_path}")
                    for k in f.keys():
                        weight_dict[k] = f.get_tensor(k)
                        if weight_dict[k].dtype == torch.float:
                            weight_dict[k] = weight_dict[k].to(torch.bfloat16)

        return weight_dict

    def _init_weights(self, weight_dict=None):
        if weight_dict is None:
            if GET_RUNNING_FLAG() == "save_naive_quant" or self.config["mm_config"].get("weight_auto_quant", False) or self.config["mm_config"].get("mm_type", "Default") == "Default":
                self.original_weight_dict = self._load_ckpt()
            else:
                self.original_weight_dict = self._load_quant_ckpt()
        else:
            self.original_weight_dict = weight_dict
        # init weights
        self.pre_weight = self.pre_weight_class(self.config)
        self.post_weight = self.post_weight_class(self.config)
        self.transformer_weights = self.transformer_weight_class(self.config)
        # load weights
        self.pre_weight.load(self.original_weight_dict)
        self.post_weight.load(self.original_weight_dict)
        self.transformer_weights.load(self.original_weight_dict)

    def _init_infer(self):
        self.pre_infer = self.pre_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)
        self.transformer_infer = self.transformer_infer_class(self.config)

    def save_weights(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        pre_state_dict = self.pre_weight.state_dict()
        logger.info(pre_state_dict.keys())

        post_state_dict = self.post_weight.state_dict()
        logger.info(post_state_dict.keys())

        transformer_state_dict = self.transformer_weights.state_dict()
        logger.info(transformer_state_dict.keys())

        save_dict = {}
        save_dict.update(pre_state_dict)
        save_dict.update(post_state_dict)
        save_dict.update(transformer_state_dict)

        save_path = os.path.join(save_path, "quant_weights.pth")
        torch.save(save_dict, save_path)
        logger.info(f"Save weights to {save_path}")

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
        x = self.transformer_infer.infer(self.transformer_weights, embed, grid_sizes, *pre_infer_out)
        noise_pred_cond = self.post_infer.infer(self.post_weight, x, embed, grid_sizes)[0]

        self.scheduler.noise_pred = noise_pred_cond

        if self.config["enable_cfg"]:
            embed, grid_sizes, pre_infer_out = self.pre_infer.infer(self.pre_weight, inputs, positive=False)
            x = self.transformer_infer.infer(self.transformer_weights, embed, grid_sizes, *pre_infer_out)
            noise_pred_uncond = self.post_infer.infer(self.post_weight, x, embed, grid_sizes)[0]

            self.scheduler.noise_pred = noise_pred_uncond + self.config.sample_guide_scale * (noise_pred_cond - noise_pred_uncond)

            if self.config["cpu_offload"]:
                self.pre_weight.to_cpu()
                self.post_weight.to_cpu()
