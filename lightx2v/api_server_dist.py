import asyncio
import argparse
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from loguru import logger
import uvicorn
import threading
import ctypes
import gc
import torch
import os
import sys
import time
import torch.multiprocessing as mp
import queue
import torch.distributed as dist
import random

from lightx2v.utils.set_config import set_config
from lightx2v.infer import init_runner
from lightx2v.utils.service_utils import TaskStatusMessage, BaseServiceStatus, ProcessManager
import httpx
from pathlib import Path
from urllib.parse import urlparse

# =========================
# FastAPI Related Code
# =========================

runner = None
thread = None

app = FastAPI()


INPUT_IMAGE_DIR = Path(__file__).parent / "assets" / "inputs" / "imgs"
OUTPUT_VIDEO_DIR = Path(__file__).parent / "save_results"
INPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)


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


def download_image(image_url: str):
    with httpx.Client(verify=False) as client:
        response = client.get(image_url)

    image_name = Path(urlparse(image_url).path).name
    if not image_name:
        raise ValueError(f"Invalid image URL: {image_url}")

    image_path = INPUT_IMAGE_DIR / image_name
    image_path.parent.mkdir(parents=True, exist_ok=True)

    if response.status_code == 200:
        with open(image_path, "wb") as f:
            f.write(response.content)
        return image_path
    else:
        raise ValueError(f"Failed to download image from {image_url}")


def local_video_generate(message: Message):
    try:
        global input_queues, output_queues

        if input_queues is None or output_queues is None:
            logger.error("分布式推理服务未启动")
            ApiServerServiceStatus.record_failed_task(message, error="分布式推理服务未启动")
            return

        logger.info(f"提交任务到分布式推理服务: {message.task_id}")

        # 将任务数据转换为字典
        task_data = {
            "task_id": message.task_id,
            "prompt": message.prompt,
            "use_prompt_enhancer": message.use_prompt_enhancer,
            "negative_prompt": message.negative_prompt,
            "image_path": message.image_path,
            "num_fragments": message.num_fragments,
            "save_video_path": message.save_video_path,
        }

        if message.image_path.startswith("http"):
            image_path = download_image(message.image_path)
            task_data["image_path"] = str(image_path)

        save_video_path = Path(message.save_video_path)
        if not save_video_path.is_absolute():
            task_data["save_video_path"] = str(OUTPUT_VIDEO_DIR / message.save_video_path)

        # 将任务放入输入队列
        for input_queue in input_queues:
            input_queue.put(task_data)

        # 等待结果
        timeout = 300  # 5分钟超时
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                result = output_queues[0].get(timeout=1.0)

                # 检查是否是当前任务的结果
                if result.get("task_id") == message.task_id:
                    if result.get("status") == "success":
                        logger.info(f"任务 {message.task_id} 推理成功")
                        ApiServerServiceStatus.complete_task(message)
                    else:
                        error_msg = result.get("error", "推理失败")
                        logger.error(f"任务 {message.task_id} 推理失败: {error_msg}")
                        ApiServerServiceStatus.record_failed_task(message, error=error_msg)
                    return
                else:
                    # 不是当前任务的结果，放回队列
                    output_queues[0].put(result)
                    time.sleep(0.1)

            except queue.Empty:
                # 队列为空，继续等待
                continue

        # 超时
        logger.error(f"任务 {message.task_id} 处理超时")
        ApiServerServiceStatus.record_failed_task(message, error="处理超时")

    except Exception as e:
        logger.error(f"任务 {message.task_id} 处理失败: {str(e)}")
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


@app.get("/v1/local/video/generate/get_task_result")
async def get_task_result(message: TaskStatusMessage):
    result = ApiServerServiceStatus.get_status_task_id(message.task_id)
    # 传输save_video_path内容到外部
    save_video_path = result.get("save_video_path")

    if save_video_path and Path(save_video_path).is_absolute() and Path(save_video_path).exists():
        return FileResponse(save_video_path)
    elif save_video_path and not Path(save_video_path).is_absolute():
        video_path = OUTPUT_VIDEO_DIR / save_video_path
        if video_path.exists():
            return FileResponse(video_path)

    return {"status": "not_found", "message": "Task result not found"}


@app.get("/v1/file/download")
async def download_file(file_path: str):
    try:
        full_path = OUTPUT_VIDEO_DIR / file_path
        resolved_path = full_path.resolve()
        if OUTPUT_VIDEO_DIR not in resolved_path.parents and resolved_path != OUTPUT_VIDEO_DIR:
            logger.warning(f"检测到路径遍历尝试：{file_path} 尝试访问 {resolved_path}")
            return {"status": "forbidden", "message": "不允许访问指定路径之外的文件"}

        if resolved_path.exists() and resolved_path.is_file():
            return FileResponse(resolved_path)
        else:
            return {"status": "not_found", "message": f"文件未找到: {file_path}"}
    except Exception as e:
        logger.error(f"处理文件下载请求时发生错误: {e}")
        return {"status": "error", "message": "文件下载失败"}


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


# 使用多进程队列进行通信
input_queues = []
output_queues = []
distributed_runners = []


def distributed_inference_worker(rank, world_size, master_addr, master_port, args, input_queue, output_queue):
    """分布式推理服务工作进程"""
    try:
        # 设置环境变量
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["ENABLE_PROFILING_DEBUG"] = "true"
        os.environ["ENABLE_GRAPH_MODE"] = "false"
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

        logger.info(f"进程 {rank}/{world_size - 1} 正在初始化分布式推理服务...")

        dist.init_process_group(backend="nccl", init_method=f"tcp://{master_addr}:{master_port}", rank=rank, world_size=world_size)

        config = set_config(args)
        config["mode"] = "server"
        logger.info(f"config: {config}")
        runner = init_runner(config)

        logger.info(f"进程 {rank}/{world_size - 1} 分布式推理服务初始化完成，等待任务...")

        while True:
            try:
                task_data = input_queue.get(timeout=1.0)  # 1秒超时
                if task_data is None:  # 停止信号
                    logger.info(f"进程 {rank}/{world_size - 1} 收到停止信号，退出推理服务")
                    break
                logger.info(f"进程 {rank}/{world_size - 1} 收到推理任务: {task_data['task_id']}")

                runner.set_inputs(task_data)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # 运行推理，复用已创建的事件循环
                try:
                    loop.run_until_complete(runner.run_pipeline())

                    # 只有 Rank 0 负责将结果放入输出队列，避免重复
                    if rank == 0:
                        result = {"task_id": task_data["task_id"], "status": "success", "save_video_path": task_data["save_video_path"], "message": "推理完成"}
                        output_queue.put(result)
                        logger.info(f"任务 {task_data['task_id']} 处理完成 (由 Rank 0 报告)")
                    dist.barrier()

                except Exception as e:
                    # 只有 Rank 0 负责报告错误
                    if rank == 0:
                        result = {"task_id": task_data["task_id"], "status": "failed", "error": str(e), "message": f"推理失败: {str(e)}"}
                        output_queue.put(result)
                        logger.error(f"任务 {task_data['task_id']} 推理失败: {str(e)} (由 Rank 0 报告)")
                    dist.barrier()

            except queue.Empty:
                # 队列为空，继续等待
                continue
            except Exception as e:
                logger.error(f"进程 {rank}/{world_size - 1} 处理任务时发生错误: {str(e)}")
                # 只有 Rank 0 负责发送错误结果
                if rank == 0:
                    error_result = {
                        "task_id": task_data.get("task_id", "unknown") if "task_data" in locals() else "unknown",
                        "status": "error",
                        "error": str(e),
                        "message": f"处理任务时发生错误: {str(e)}",
                    }
                    output_queue.put(error_result)
                dist.barrier()

    except Exception as e:
        logger.error(f"分布式推理服务进程 {rank}/{world_size - 1} 启动失败: {str(e)}")
        # 只有 Rank 0 负责报告启动失败
        if rank == 0:
            error_result = {"task_id": "startup", "status": "startup_failed", "error": str(e), "message": f"推理服务启动失败: {str(e)}"}
            output_queue.put(error_result)
    # 在进程最终退出时关闭事件循环和销毁分布式组
    finally:
        if "loop" in locals() and loop and not loop.is_closed():
            loop.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def start_distributed_inference_with_queue(args):
    """使用队列启动分布式推理服务，并模拟torchrun的多进程模式"""
    global input_queues, output_queues, distributed_runners

    nproc_per_node = args.nproc_per_node

    if nproc_per_node <= 0:
        logger.error("nproc_per_node 必须大于0")
        return False

    try:
        master_addr = "127.0.0.1"
        master_port = str(random.randint(20000, 29999))
        logger.info(f"分布式推理服务 Master Addr: {master_addr}, Master Port: {master_port}")

        # 创建队列
        ctx = mp.get_context("spawn")
        # 使用spawn启动多进程
        processes = []
        for rank in range(nproc_per_node):
            input_queue = ctx.Queue()
            output_queue = ctx.Queue()
            p = ctx.Process(target=distributed_inference_worker, args=(rank, nproc_per_node, master_addr, master_port, args, input_queue, output_queue), daemon=True)

            p.start()
            processes.append(p)
            input_queues.append(input_queue)
            output_queues.append(output_queue)

        distributed_runners = processes
        return True

    except Exception as e:
        logger.exception(f"启动分布式推理服务时发生错误: {str(e)}")
        stop_distributed_inference_with_queue()
        return False


def stop_distributed_inference_with_queue():
    """停止分布式推理服务"""
    global input_queues, output_queues, distributed_runners

    try:
        if distributed_runners:
            logger.info(f"正在停止 {len(distributed_runners)} 个分布式推理服务进程...")

            # 向所有工作进程发送停止信号
            if input_queues:
                for input_queue in input_queues:
                    input_queue.put(None)

            # 等待所有进程结束
            for p in distributed_runners:
                p.join(timeout=10)

            # 强制终止任何未结束的进程
            for p in distributed_runners:
                if p.is_alive():
                    logger.warning(f"推理服务进程 {p.pid} 未在规定时间内结束，强制终止...")
                    p.terminate()
                    p.join()

            logger.info("所有分布式推理服务进程已停止")

        # 清理队列
        if input_queues:
            try:
                for input_queue in input_queues:
                    while not input_queue.empty():
                        input_queue.get_nowait()
            except:  # noqa: E722
                pass

        if output_queues:
            try:
                for output_queue in output_queues:
                    while not output_queue.empty():
                        output_queue.get_nowait()
            except:  # noqa: E722
                pass

        distributed_runners = []
        input_queue = None
        output_queue = None

    except Exception as e:
        logger.error(f"停止分布式推理服务时发生错误: {str(e)}")


# =========================
# Main Entry
# =========================

if __name__ == "__main__":
    global startup_args

    ProcessManager.register_signal_handler()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cls", type=str, required=True, choices=["wan2.1", "hunyuan", "wan2.1_causvid", "wan2.1_skyreels_v2_df"], default="hunyuan")
    parser.add_argument("--task", type=str, choices=["t2v", "i2v"], default="t2v")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_json", type=str, required=True)
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--start_inference", action="store_true", help="是否在启动API服务器前启动分布式推理服务")
    parser.add_argument("--nproc_per_node", type=int, default=4, help="分布式推理时每个节点的进程数")

    args = parser.parse_args()
    logger.info(f"args: {args}")

    # 保存启动参数供重启功能使用
    startup_args = args

    if args.start_inference:
        logger.info("正在启动分布式推理服务...")
        success = start_distributed_inference_with_queue(args)
        if not success:
            logger.error("分布式推理服务启动失败，退出程序")
            sys.exit(1)

        # 注册程序退出时的清理函数
        import atexit

        atexit.register(stop_distributed_inference_with_queue)

        # 注册信号处理器，用于优雅关闭
        import signal

        def signal_handler(signum, frame):
            logger.info(f"接收到信号 {signum}，正在优雅关闭...")
            stop_distributed_inference_with_queue()
            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    try:
        logger.info(f"正在启动FastAPI服务器，端口: {args.port}")
        uvicorn.run(app, host="0.0.0.0", port=args.port, reload=False, workers=1)
    except KeyboardInterrupt:
        logger.info("接收到KeyboardInterrupt，正在关闭服务...")
    except Exception as e:
        logger.error(f"FastAPI服务器运行时发生错误: {str(e)}")
    finally:
        # 确保在程序结束时停止推理服务
        if args.start_inference:
            stop_distributed_inference_with_queue()
