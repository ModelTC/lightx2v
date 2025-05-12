import json
import os
from easydict import EasyDict


def get_default_config():
    default_config = {
        "do_mm_calib": False,
        "cpu_offload": False,
        "parallel_attn_type": None,  # [None, "ulysses", "ring"]
        "parallel_vae": False,
        "max_area": False,
        "vae_stride": (4, 8, 8),
        "patch_size": (1, 2, 2),
        "feature_caching": "NoCaching",  # ["NoCaching", "TaylorSeer", "Tea"]
        "teacache_thresh": 0.26,
        "use_ret_steps": False,
        "use_bfloat16": True,
        "lora_path": None,
        "strength_model": 1.0,
        "mm_config": {},
        "use_prompt_enhancer": False,
    }
    return default_config


def set_config(args):
    config = get_default_config()
    config.update({k: v for k, v in vars(args).items()})
    config = EasyDict(config)

    with open(config.config_json, "r") as f:
        config_json = json.load(f)
    config.update(config_json)

    if os.path.exists(os.path.join(config.model_path, "config.json")):
        with open(os.path.join(config.model_path, "config.json"), "r") as f:
            model_config = json.load(f)
        config.update(model_config)

    return config
