import signal
import sys
import psutil
import argparse
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
import json
import torch
import gc
import os
import asyncio

from lightx2v.utils.profiler import ProfilingContext
from lightx2v.utils.set_config import set_config
from lightx2v.infer import init_runner
from lightx2v.utils.prompt_enhancer import PromptEnhancer


# =========================
# Signal & Process Control
# =========================


def kill_all_related_processes():
    """Kill the current process and all its child processes"""
    current_process = psutil.Process()
    children = current_process.children(recursive=True)
    for child in children:
        try:
            child.kill()
        except Exception as e:
            print(f"Failed to kill child process {child.pid}: {e}")
    try:
        current_process.kill()
    except Exception as e:
        print(f"Failed to kill main process: {e}")


def signal_handler(sig, frame):
    print("\nReceived Ctrl+C, shutting down all related processes...")
    kill_all_related_processes()
    sys.exit(0)


# =========================
# FastAPI Related Code
# =========================

runner = None

app = FastAPI()


class Message(BaseModel):
    prompt: str
    negative_prompt: str = ""
    image_path: str = ""
    save_video_path: str
    use_prompt_enhancer: bool = False

    def get(self, key, default=None):
        return getattr(self, key, default)


@app.post("/v1/local/video/generate")
async def v1_local_video_generate(message: Message, request: Request):
    if message.use_prompt_enhancer and prompt_enhancer is not None:
        with ProfilingContext("Prompt Enhancer Cost"):
            print(f"Enhancing prompt using model")
            enhanced_prompt = prompt_enhancer(message.prompt)
            print(f"Original prompt: {message.prompt}")
            print(f"Enhanced prompt: {enhanced_prompt}")
            message.prompt = enhanced_prompt

    global runner
    runner.set_inputs(message)
    await asyncio.to_thread(runner.run_pipeline)
    return {"response": "finished", "prompt": message.prompt, "save_video_path": message.save_video_path}


# =========================
# Main Entry
# =========================

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cls", type=str, required=True, choices=["wan2.1", "hunyuan", "wan2.1_causal"], default="hunyuan")
    parser.add_argument("--task", type=str, choices=["t2v", "i2v"], default="t2v")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_json", type=str, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt_enhancer", type=str, nargs="?", const="Qwen/Qwen2.5-32B-Instruct", help="Enable prompt enhancer with optional model name if GPU count >= 2")
    args = parser.parse_args()
    print(f"args: {args}")

    gpu_count = torch.cuda.device_count()
    print(f"Available GPU count: {gpu_count}")

    prompt_enhancer, enhancer_gpu_id = None, None

    # Only enable prompt_enhancer if we have enough GPUs and the argument is provided
    if args.prompt_enhancer is not None:
        if gpu_count >= 2:
            enhancer_gpu_id = gpu_count - 1  # Use the last GPU for enhancer
            with ProfilingContext("Init Prompt Enhancer Cost"):
                print(f"Initializing prompt enhancer on cuda:{enhancer_gpu_id}")
                prompt_enhancer = PromptEnhancer(model_name=args.prompt_enhancer, device_map=f"cuda:{enhancer_gpu_id}")
        else:
            print("Warning: prompt_enhancer requires at least 2 GPUs, disabling this feature")

    with ProfilingContext("Init Server Cost"):
        if gpu_count >= 2 and enhancer_gpu_id is not None:
            video_gpus = list(range(enhancer_gpu_id))  # All GPUs except the enhancer GPU
            video_gpus_str = ",".join(map(str, video_gpus))
            print(f"Setting video model to use GPUs: {video_gpus_str}")

            original_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            os.environ["CUDA_VISIBLE_DEVICES"] = video_gpus_str

            # Initialize the runner with only video GPUs visible
            config = set_config(args)
            print(f"config:\n{json.dumps(config, ensure_ascii=False, indent=4)}")
            runner = init_runner(config)

            # Restore original CUDA_VISIBLE_DEVICES
            if original_cuda_devices:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_devices
            else:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            # No need to restrict GPUs
            config = set_config(args)
            print(f"config:\n{json.dumps(config, ensure_ascii=False, indent=4)}")
            runner = init_runner(config)

    uvicorn.run(app, host="0.0.0.0", port=config.port, reload=False, workers=1)
