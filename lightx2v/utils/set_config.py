import json
import os
from easydict import EasyDict


def set_config(args):
    config = {k: v for k, v in vars(args).items()}
    config = EasyDict(config)

    if args.mm_config:
        config.mm_config = json.loads(args.mm_config)
    else:
        config.mm_config = None

    with open(os.path.join(args.model_path, "config.json"), "r") as f:
        model_config = json.load(f)
    config.update(model_config)

    return config
