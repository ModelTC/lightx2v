import os
import torch
from safetensors import safe_open
from loguru import logger
import gc


class WanLoraWrapper:
    def __init__(self, wan_model):
        self.model = wan_model
        self.lora_dict = {}
        self.override_dict = {}

    def load_lora(self, lora_path, lora_name=None):
        if lora_name is None:
            lora_name = os.path.basename(lora_path).split(".")[0]

        if lora_name in self.lora_dict:
            logger.info(f"LoRA {lora_name} already loaded, skipping...")
            return lora_name

        lora_weights = self._load_lora_file(lora_path)

        self.lora_dict[lora_name] = lora_weights
        return lora_name

    def _load_lora_file(self, file_path):
        use_bfloat16 = True  # Default value
        if self.model.config and hasattr(self.model.config, "get"):
            use_bfloat16 = self.model.config.get("use_bfloat16", True)
        with safe_open(file_path, framework="pt") as f:
            if use_bfloat16:
                tensor_dict = {key: f.get_tensor(key).to(torch.bfloat16) for key in f.keys()}
            else:
                tensor_dict = {key: f.get_tensor(key) for key in f.keys()}
        return tensor_dict

    def apply_lora(self, lora_name, alpha=1.0):
        if lora_name not in self.lora_dict:
            logger.info(f"LoRA {lora_name} not found. Please load it first.")

        if hasattr(self.model, "current_lora") and self.model.current_lora:
            self.remove_lora()

        if not hasattr(self.model, "original_weight_dict"):
            logger.error("Model does not have 'original_weight_dict'. Cannot apply LoRA.")
            return False

        weight_dict = self.model.original_weight_dict
        lora_weights = self.lora_dict[lora_name]
        self._apply_lora_weights(weight_dict, lora_weights, alpha)

        # 重新加载权重
        self.model.pre_weight.load_weights(weight_dict)
        self.model.post_weight.load_weights(weight_dict)
        self.model.transformer_weights.load_weights(weight_dict)

        self.model.current_lora = lora_name
        logger.info(f"Applied LoRA: {lora_name} with alpha={alpha}")
        return True

    def _apply_lora_weights(self, weight_dict, lora_weights, alpha):
        lora_pairs = {}
        prefix = "diffusion_model."

        for key in lora_weights.keys():
            if key.endswith("lora_A.weight") and key.startswith(prefix):
                base_name = key[len(prefix) :].replace("lora_A.weight", "weight")
                b_key = key.replace("lora_A.weight", "lora_B.weight")
                if b_key in lora_weights:
                    lora_pairs[base_name] = (key, b_key)

        applied_count = 0
        for name, param in weight_dict.items():
            if name in lora_pairs:
                name_lora_A, name_lora_B = lora_pairs[name]
                lora_A = lora_weights[name_lora_A].to(param.device, param.dtype)
                lora_B = lora_weights[name_lora_B].to(param.device, param.dtype)
                param += torch.matmul(lora_B, lora_A) * alpha
                applied_count += 1

        logger.info(f"Applied {applied_count} LoRA weight adjustments")
        if applied_count == 0:
            logger.info(
                "Warning: No LoRA weights were applied. Expected naming conventions: 'diffusion_model.<layer_name>.lora_A.weight' and 'diffusion_model.<layer_name>.lora_B.weight'. Please verify the LoRA weight file."
            )

    def remove_lora(self):
        if not self.model.current_lora:
            logger.info("No LoRA currently applied")
            return
        logger.info(f"Removing LoRA {self.model.current_lora}...")

        restored_count = 0
        for k, v in self.override_dict.items():
            self.model.original_weight_dict[k] = v.to(self.model.device)
            restored_count += 1

        logger.info(f"LoRA {self.model.current_lora} removed, restored {restored_count} weights")

        self.model.pre_weight.load_weights(self.model.original_weight_dict)
        self.model.post_weight.load_weights(self.model.original_weight_dict)
        self.model.transformer_weights.load_weights(self.model.original_weight_dict)

        if self.model.current_lora and self.model.current_lora in self.lora_dict:
            del self.lora_dict[self.model.current_lora]
        self.override_dict = {}

        torch.cuda.empty_cache()
        gc.collect()

    def list_loaded_loras(self):
        return list(self.lora_dict.keys())

    def get_current_lora(self):
        return self.model.current_lora
