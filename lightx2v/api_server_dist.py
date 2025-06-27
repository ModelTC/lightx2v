import asyncio
import argparse
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
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
import uuid

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

CACHE_DIR = Path(__file__).parent.parent / "cache"
INPUT_IMAGE_DIR = CACHE_DIR / "inputs" / "imgs"
OUTPUT_VIDEO_DIR = CACHE_DIR / "outputs"

for directory in [INPUT_IMAGE_DIR, OUTPUT_VIDEO_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


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


stop_generation_event = threading.Event()


def local_video_generate(message: Message, stop_event: threading.Event):
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
            if stop_event.is_set():
                logger.info(f"任务 {message.task_id} 收到停止信号，正在终止")
                ApiServerServiceStatus.record_failed_task(message, error="任务被停止")
                return

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
                    # 注意：如果并发任务很多，这种做法可能导致当前任务的结果被延迟。
                    # 更健壮的并发结果处理需要更复杂的设计，例如每个任务有独立的输出队列。
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
        global thread, stop_generation_event
        stop_generation_event.clear()
        thread = threading.Thread(
            target=local_video_generate,
            args=(
                message,
                stop_generation_event,
            ),
            daemon=True,
        )
        thread.start()
        return {"task_id": task_id, "task_status": "processing", "save_video_path": message.save_video_path}
    except RuntimeError as e:
        return {"error": str(e)}


@app.post("/v1/local/video/generate_form")
async def v1_local_video_generate_form(
    task_id: str,
    prompt: str,
    save_video_path: str,
    task_id_must_unique: bool = False,
    use_prompt_enhancer: bool = False,
    negative_prompt: str = "",
    num_fragments: int = 1,
    image_file: UploadFile = File(None),
):
    # 处理上传的图片文件
    image_path = ""
    if image_file and image_file.filename:
        # 生成唯一的文件名
        file_extension = Path(image_file.filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        image_path = INPUT_IMAGE_DIR / unique_filename

        # 保存上传的文件
        with open(image_path, "wb") as buffer:
            content = await image_file.read()
            buffer.write(content)

        image_path = str(image_path)

    message = Message(
        task_id=task_id,
        task_id_must_unique=task_id_must_unique,
        prompt=prompt,
        use_prompt_enhancer=use_prompt_enhancer,
        negative_prompt=negative_prompt,
        image_path=image_path,
        num_fragments=num_fragments,
        save_video_path=save_video_path,
    )
    try:
        task_id = ApiServerServiceStatus.start_task(message)
        global thread, stop_generation_event
        stop_generation_event.clear()
        thread = threading.Thread(
            target=local_video_generate,
            args=(
                message,
                stop_generation_event,
            ),
            daemon=True,
        )
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
        file_path = Path(save_video_path)
        relative_path = file_path.relative_to(OUTPUT_VIDEO_DIR.resolve()) if str(file_path).startswith(str(OUTPUT_VIDEO_DIR.resolve())) else file_path.name
        return {
            "status": "success",
            "task_status": result.get("status", "unknown"),
            "filename": file_path.name,
            "file_size": file_path.stat().st_size,
            "download_url": f"/v1/file/download/{relative_path}",
            "message": "任务结果已准备就绪",
        }
    elif save_video_path and not Path(save_video_path).is_absolute():
        video_path = OUTPUT_VIDEO_DIR / save_video_path
        if video_path.exists():
            return {
                "status": "success",
                "task_status": result.get("status", "unknown"),
                "filename": video_path.name,
                "file_size": video_path.stat().st_size,
                "download_url": f"/v1/file/download/{save_video_path}",
                "message": "任务结果已准备就绪",
            }

    return {"status": "not_found", "message": "Task result not found", "task_status": result.get("status", "unknown")}


def file_stream_generator(file_path: str, chunk_size: int = 1024 * 1024):
    """文件流生成器，逐块读取文件"""
    with open(file_path, "rb") as file:
        while chunk := file.read(chunk_size):
            yield chunk


@app.get(
    "/v1/file/download/{file_path:path}",
    response_class=StreamingResponse,
    summary="下载文件",
    description="流式下载指定的文件",
    responses={200: {"description": "文件下载成功", "content": {"application/octet-stream": {}}}, 404: {"description": "文件未找到"}, 500: {"description": "服务器错误"}},
)
async def download_file(file_path: str):
    try:
        full_path = OUTPUT_VIDEO_DIR / file_path
        resolved_path = full_path.resolve()

        # 安全检查：确保文件在允许的目录内
        if not str(resolved_path).startswith(str(OUTPUT_VIDEO_DIR.resolve())):
            return {"status": "forbidden", "message": "不允许访问该文件"}

        if resolved_path.exists() and resolved_path.is_file():
            file_size = resolved_path.stat().st_size
            filename = resolved_path.name

            # 设置适当的 MIME 类型
            mime_type = "application/octet-stream"
            if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                mime_type = "video/mp4"
            elif filename.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
                mime_type = "image/jpeg"

            headers = {
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(file_size),
                "Accept-Ranges": "bytes",
            }

            return StreamingResponse(file_stream_generator(str(resolved_path)), media_type=mime_type, headers=headers)
        else:
            return {"status": "not_found", "message": f"文件未找到: {file_path}"}
    except Exception as e:
        logger.error(f"处理文件下载请求时发生错误: {e}")
        return {"status": "error", "message": "文件下载失败"}


@app.get("/v1/local/video/generate/stop_running_task")
async def stop_running_task():
    global thread, stop_generation_event
    if thread and thread.is_alive():
        try:
            logger.info("正在发送停止信号给运行中的任务线程...")
            stop_generation_event.set()  # 设置事件，通知线程停止
            thread.join(timeout=5)  # 等待线程结束，设置超时时间

            if thread.is_alive():
                logger.warning("任务线程未在规定时间内停止，可能需要手动干预。")
                return {"stop_status": "warning", "reason": "任务线程未在规定时间内停止，可能需要手动干预。"}
            else:
                # 清理线程引用
                thread = None
                ApiServerServiceStatus.clean_stopped_task()
                gc.collect()
                torch.cuda.empty_cache()
                logger.info("任务已成功停止。")
                return {"stop_status": "success", "reason": "Task stopped successfully."}
        except Exception as e:
            logger.error(f"停止任务时发生错误: {str(e)}")
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

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while True:
            try:
                task_data = input_queue.get(timeout=1.0)  # 1秒超时
                if task_data is None:  # 停止信号
                    logger.info(f"进程 {rank}/{world_size - 1} 收到停止信号，退出推理服务")
                    break
                logger.info(f"进程 {rank}/{world_size - 1} 收到推理任务: {task_data['task_id']}")

                runner.set_inputs(task_data)

                # 运行推理，复用已创建的事件循环
                try:
                    loop.run_until_complete(runner.run_pipeline())

                    # 只有 Rank 0 负责将结果放入输出队列，避免重复
                    if rank == 0:
                        result = {"task_id": task_data["task_id"], "status": "success", "save_video_path": task_data["save_video_path"], "message": "推理完成"}
                        output_queue.put(result)
                        logger.info(f"任务 {task_data['task_id']} 处理完成 (由 Rank 0 报告)")
                    if dist.is_initialized():
                        dist.barrier()

                except Exception as e:
                    # 只有 Rank 0 负责报告错误
                    if rank == 0:
                        result = {"task_id": task_data["task_id"], "status": "failed", "error": str(e), "message": f"推理失败: {str(e)}"}
                        output_queue.put(result)
                        logger.error(f"任务 {task_data['task_id']} 推理失败: {str(e)} (由 Rank 0 报告)")
                    if dist.is_initialized():
                        dist.barrier()

            except queue.Empty:
                # 队列为空，继续等待
                continue
            except KeyboardInterrupt:
                logger.info(f"进程 {rank}/{world_size - 1} 收到 KeyboardInterrupt，优雅退出")
                break
            except Exception as e:
                logger.error(f"进程 {rank}/{world_size - 1} 处理任务时发生错误: {str(e)}")
                # 只有 Rank 0 负责发送错误结果
                task_data = task_data if "task_data" in locals() else {}
                if rank == 0:
                    error_result = {
                        "task_id": task_data.get("task_id", "unknown"),
                        "status": "error",
                        "error": str(e),
                        "message": f"处理任务时发生错误: {str(e)}",
                    }
                    try:
                        output_queue.put(error_result)
                    except:  # noqa: E722
                        pass
                if dist.is_initialized():
                    try:
                        dist.barrier()
                    except:  # noqa: E722
                        pass

    except KeyboardInterrupt:
        logger.info(f"进程 {rank}/{world_size - 1} 主循环收到 KeyboardInterrupt，正在退出")
    except Exception as e:
        logger.error(f"分布式推理服务进程 {rank}/{world_size - 1} 启动失败: {str(e)}")
        # 只有 Rank 0 负责报告启动失败
        if rank == 0:
            try:
                error_result = {"task_id": "startup", "status": "startup_failed", "error": str(e), "message": f"推理服务启动失败: {str(e)}"}
                output_queue.put(error_result)
            except:  # noqa: E722
                pass
    # 在进程最终退出时关闭事件循环和销毁分布式组
    finally:
        try:
            if "loop" in locals() and loop and not loop.is_closed():
                loop.close()
        except:  # noqa: E722
            pass
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except:  # noqa: E722
            pass


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

        processes = []

        ctx = mp.get_context("spawn")
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
                    try:
                        input_queue.put(None)
                    except:  # noqa: E722
                        pass

            # 等待所有进程结束
            for p in distributed_runners:
                try:
                    p.join(timeout=10)
                except:  # noqa: E722
                    pass

            # 强制终止任何未结束的进程
            for p in distributed_runners:
                try:
                    if p.is_alive():
                        logger.warning(f"推理服务进程 {p.pid} 未在规定时间内结束，强制终止...")
                        p.terminate()
                        p.join(timeout=5)
                except:  # noqa: E722
                    pass

            logger.info("所有分布式推理服务进程已停止")

        # 清理队列
        if input_queues:
            try:
                for input_queue in input_queues:
                    try:
                        while not input_queue.empty():
                            input_queue.get_nowait()
                    except:  # noqa: E722
                        pass
            except:  # noqa: E722
                pass

        if output_queues:
            try:
                for output_queue in output_queues:
                    try:
                        while not output_queue.empty():
                            output_queue.get_nowait()
                    except:  # noqa: E722
                        pass
            except:  # noqa: E722
                pass

        distributed_runners = []
        input_queues = []
        output_queues = []

    except Exception as e:
        logger.error(f"停止分布式推理服务时发生错误: {str(e)}")
    except KeyboardInterrupt:
        logger.info("停止分布式推理服务时收到 KeyboardInterrupt，强制清理")


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
            try:
                stop_distributed_inference_with_queue()
            except:  # noqa: E722
                logger.error("关闭分布式推理服务时发生错误")
            finally:
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

"""
curl -X 'POST' \
  'http://localhost:8000/v1/local/video/generate_form?task_id=abc&prompt=%E8%B7%B3%E8%88%9E&save_video_path=a.mp4&task_id_must_unique=false&use_prompt_enhancer=false&num_fragments=1' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image_file=@图片1.png;type=image/png'

curl -X 'POST' \
  'http://localhost:8000/v1/local/video/generate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "task_id": "abcde",
  "task_id_must_unique": false,
  "prompt": "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline'\''s intricate details and the refreshing atmosphere of the seaside.",
  "use_prompt_enhancer": false,
  "negative_prompt": "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
  "image_path": "/mnt/aigc/users/gaopeng1/ComfyUI-Lightx2vWrapper/lightx2v/assets/inputs/imgs/img_0.jpg",
  "num_fragments": 1,
  "save_video_path": "/mnt/aigc/users/lijiaqi2/ComfyUI/custom_nodes/ComfyUI-Lightx2vWrapper/lightx2v/save_results/img_0.mp4"
}'

"""
