import asyncio
import argparse
from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger
import uvicorn
import json
from typing import Optional
from datetime import datetime
import threading
import ctypes
import gc
import torch

from lightx2v.utils.profiler import ProfilingContext
from lightx2v.utils.set_config import set_config
from lightx2v.infer import init_runner
from lightx2v.utils.service_utils import TaskStatusMessage, BaseServiceStatus, ProcessManager


# =========================
# FastAPI Related Code
# =========================

runner = None
thread = None

app = FastAPI()


class Message(BaseModel):
    task_id: str
    task_id_must_unique: bool = False

    prompt: str
    use_prompt_enhancer: bool = False
    negative_prompt: str = ""
    image_path: str = ""
    num_fragments: int = 1
    save_video_path: str

    def get(self, key, default=None):
        return getattr(self, key, default)


class ApiServerServiceStatus(BaseServiceStatus):
    pass


def local_video_generate(message: Message):
    try:
        global runner
        runner.set_inputs(message)
        logger.info(f"message: {message}")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(runner.run_pipeline())
        finally:
            loop.close()
        ApiServerServiceStatus.complete_task(message)
    except Exception as e:
        logger.error(f"task_id {message.task_id} failed: {str(e)}")
        ApiServerServiceStatus.record_failed_task(message, error=str(e))


@app.post("/v1/local/video/generate")
async def v1_local_video_generate(message: Message):
    try:
        task_id = ApiServerServiceStatus.start_task(message)
        # Use background threads to perform long-running tasks
        global thread
        thread = threading.Thread(target=local_video_generate, args=(message,), daemon=True)
        thread.start()
        return {"task_id": task_id, "task_status": "processing", "save_video_path": message.save_video_path}
    except RuntimeError as e:
        return {"error": str(e)}


@app.get("/v1/local/video/generate/service_status")
async def get_service_status():
    return ApiServerServiceStatus.get_status_service()


@app.get("/v1/local/video/generate/get_all_tasks")
async def get_all_tasks():
    return ApiServerServiceStatus.get_all_tasks()


@app.post("/v1/local/video/generate/task_status")
async def get_task_status(message: TaskStatusMessage):
    return ApiServerServiceStatus.get_status_task_id(message.task_id)


def _async_raise(tid, exctype):
    """Force thread tid to raise exception exctype"""
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("Invalid thread ID")
    elif res > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), 0)
        raise SystemError("PyThreadState_SetAsyncExc failed")


@app.get("/v1/local/video/generate/stop_running_task")
async def stop_running_task():
    global thread
    if thread and thread.is_alive():
        try:
            _async_raise(thread.ident, SystemExit)
            thread.join()

            # Clean up the thread reference
            thread = None
            ApiServerServiceStatus.clean_stopped_task()
            gc.collect()
            torch.cuda.empty_cache()
            return {"stop_status": "success", "reason": "Task stopped successfully."}
        except Exception as e:
            return {"stop_status": "error", "reason": str(e)}
    else:
        return {"stop_status": "do_nothing", "reason": "No running task found."}


# =========================
# Main Entry
# =========================

if __name__ == "__main__":
    ProcessManager.register_signal_handler()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cls", type=str, required=True, choices=["wan2.1", "hunyuan", "wan2.1_causvid", "wan2.1_skyreels_v2_df", "cogvideox"], default="hunyuan")
    parser.add_argument("--task", type=str, choices=["t2v", "i2v"], default="t2v")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_json", type=str, required=True)
    parser.add_argument("--split", action="store_true")

    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logger.info(f"args: {args}")

    with ProfilingContext("Init Server Cost"):
        config = set_config(args)
        config["mode"] = "split_server" if args.split else "server"
        logger.info(f"config:\n{json.dumps(config, ensure_ascii=False, indent=4)}")
        runner = init_runner(config)

    uvicorn.run(app, host="0.0.0.0", port=config.port, reload=False, workers=1)
