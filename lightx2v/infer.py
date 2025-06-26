import asyncio
import argparse
import torch
import torch.distributed as dist
import json

from lightx2v.utils.envs import *
from lightx2v.utils.utils import seed_all
from lightx2v.utils.profiler import ProfilingContext
from lightx2v.utils.set_config import set_config
from lightx2v.utils.registry_factory import RUNNER_REGISTER

from lightx2v.models.runners.hunyuan.hunyuan_runner import HunyuanRunner
from lightx2v.models.runners.wan.wan_runner import WanRunner
from lightx2v.models.runners.wan.wan_distill_runner import WanDistillRunner
from lightx2v.models.runners.wan.wan_causvid_runner import WanCausVidRunner
from lightx2v.models.runners.wan.wan_skyreels_v2_df_runner import WanSkyreelsV2DFRunner
from lightx2v.models.runners.graph_runner import GraphRunner
from lightx2v.models.runners.cogvideox.cogvidex_runner import CogvideoxRunner
from lightx2v.dist.wrappers.pp.runner import PipelineParallelWanRunnerWrapper

from lightx2v.common.ops import *
from loguru import logger


def init_runner(config):
    seed_all(config.seed)

    if config.parallel_attn_type:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

    if CHECK_ENABLE_GRAPH_MODE():
        default_runner = RUNNER_REGISTER[config.model_cls](config)
        runner = GraphRunner(default_runner)
        runner.runner.init_modules()
    elif config.parallel_attn_type == "pipefusion":
        default_runner = RUNNER_REGISTER[config.model_cls](config)
        default_runner.init_modules()
        runner = PipelineParallelWanRunnerWrapper(default_runner, config)
    else:
        runner = RUNNER_REGISTER[config.model_cls](config)
        runner.init_modules()
    return runner


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cls", type=str, required=True, choices=["wan2.1", "hunyuan", "wan2.1_distill", "wan2.1_causvid", "wan2.1_skyreels_v2_df", "cogvideox"], default="hunyuan")
    parser.add_argument("--task", type=str, choices=["t2v", "i2v"], default="t2v")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_json", type=str, required=True)
    parser.add_argument("--use_prompt_enhancer", action="store_true")

    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--image_path", type=str, default="", help="The path to input image file or path for image-to-video (i2v) task")
    parser.add_argument("--save_video_path", type=str, default="./output_lightx2v.mp4", help="The path to save video path/file")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    with ProfilingContext("Total Cost"):
        config = set_config(args)
        config["mode"] = "infer"
        logger.info(f"config:\n{json.dumps(config, ensure_ascii=False, indent=4)}")
        runner = init_runner(config)

        await runner.run_pipeline()


if __name__ == "__main__":
    asyncio.run(main())
