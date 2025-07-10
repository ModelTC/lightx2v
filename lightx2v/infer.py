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
from lightx2v.models.runners.wan.wan_audio_runner import WanAudioRunner
from lightx2v.models.runners.wan.wan_skyreels_v2_df_runner import WanSkyreelsV2DFRunner
from lightx2v.models.runners.graph_runner import GraphRunner
from lightx2v.models.runners.cogvideox.cogvidex_runner import CogvideoxRunner

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
    else:
        runner = RUNNER_REGISTER[config.model_cls](config)
        runner.init_modules()
    return runner


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_cls", type=str, required=True, choices=["wan2.1", "hunyuan", "wan2.1_distill", "wan2.1_causvid", "wan2.1_skyreels_v2_df", "cogvideox", "wan2.1_audio"], default="hunyuan"
    )
    parser.add_argument("--task", type=str, choices=["t2v", "i2v"], default="t2v")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_json", type=str, required=True)
    parser.add_argument("--use_prompt_enhancer", action="store_true")

    parser.add_argument("--prompt", type=str, default="", help="The input prompt for text-to-video generation")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--lora_path", type=str, default="", help="The lora file path")
    parser.add_argument("--prompt_path", type=str, default="", help="The path to input prompt file")
    parser.add_argument("--audio_path", type=str, default="", help="The path to input audio file")
    parser.add_argument("--image_path", type=str, default="", help="The path to input image file or path for image-to-video (i2v) task")
    parser.add_argument("--save_video_path", type=str, default="./output_lightx2v.mp4", help="The path to save video path/file")
    args = parser.parse_args()

    if args.prompt_path:
        try:
            with open(args.prompt_path, "r", encoding="utf-8") as f:
                args.prompt = f.read().strip()
            logger.info(f"从文件 {args.prompt_path} 读取到prompt: {args.prompt}")
        except FileNotFoundError:
            logger.error(f"找不到prompt文件: {args.prompt_path}")
            raise
        except Exception as e:
            logger.error(f"读取prompt文件时出错: {e}")
            raise

    logger.info(f"args: {args}")

    with ProfilingContext("Total Cost"):
        config = set_config(args)
        config["mode"] = "infer"
        logger.info(f"config:\n{json.dumps(config, ensure_ascii=False, indent=4)}")
        runner = init_runner(config)

        await runner.run_pipeline()


if __name__ == "__main__":
    asyncio.run(main())
