import asyncio
import queue
import time
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx
import torch.multiprocessing as mp
from loguru import logger

from ..utils.set_config import set_config
from ..infer import init_runner
from .utils import ServiceStatus
from .schema import TaskRequest, TaskResponse, TaskResultResponse
from .distributed_utils import create_distributed_worker


mp.set_start_method("spawn", force=True)


class FileService:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.input_image_dir = cache_dir / "inputs" / "imgs"
        self.input_audio_dir = cache_dir / "inputs" / "audios"
        self.output_video_dir = cache_dir / "outputs"

        # 创建目录
        for directory in [
            self.input_image_dir,
            self.output_video_dir,
            self.input_audio_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    async def download_image(self, image_url: str) -> Path:
        try:
            async with httpx.AsyncClient(verify=False) as client:
                response = await client.get(image_url)

            if response.status_code != 200:
                raise ValueError(f"Failed to download image from {image_url}")

            image_name = Path(urlparse(image_url).path).name
            if not image_name:
                raise ValueError(f"Invalid image URL: {image_url}")

            image_path = self.input_image_dir / image_name
            image_path.parent.mkdir(parents=True, exist_ok=True)

            with open(image_path, "wb") as f:
                f.write(response.content)

            return image_path
        except Exception as e:
            logger.error(f"下载图片失败: {e}")
            raise

    def save_uploaded_file(self, file_content: bytes, filename: str) -> Path:
        file_extension = Path(filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = self.input_image_dir / unique_filename

        with open(file_path, "wb") as f:
            f.write(file_content)

        return file_path

    def get_output_path(self, save_video_path: str) -> Path:
        video_path = Path(save_video_path)
        if not video_path.is_absolute():
            return self.output_video_dir / save_video_path
        return video_path


def _distributed_inference_worker(rank, world_size, master_addr, master_port, args, task_queue, result_queue):
    task_data = None
    loop = None
    worker = None

    try:
        logger.info(f"进程 {rank}/{world_size - 1} 正在初始化分布式推理服务...")

        # 创建并初始化分布式工作进程
        worker = create_distributed_worker(rank, world_size, master_addr, master_port)
        if not worker.init():
            raise RuntimeError(f"Rank {rank} 分布式环境初始化失败")

        # 初始化配置和模型
        config = set_config(args)
        config["mode"] = "server"
        logger.info(f"Rank {rank} config: {config}")

        runner = init_runner(config)
        logger.info(f"进程 {rank}/{world_size - 1} 分布式推理服务初始化完成")

        # 创建事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while True:
            # 只有rank=0从队列读取任务
            if rank == 0:
                try:
                    task_data = task_queue.get(timeout=1.0)
                    if task_data is None:  # 停止信号
                        logger.info(f"进程 {rank} 收到停止信号，退出推理服务")
                        # 广播停止信号给其他进程
                        worker.dist_manager.broadcast_task_data(None)
                        break
                    # 广播任务数据给其他进程
                    worker.dist_manager.broadcast_task_data(task_data)
                except queue.Empty:
                    # 队列为空，继续等待
                    continue
            else:
                # 非rank=0进程从rank=0接收任务数据
                task_data = worker.dist_manager.broadcast_task_data()
                if task_data is None:  # 停止信号
                    logger.info(f"进程 {rank} 收到停止信号，退出推理服务")
                    break

            # 所有进程都处理任务
            if task_data is not None:
                logger.info(f"进程 {rank} 收到推理任务: {task_data['task_id']}")

                try:
                    # 设置输入并运行推理
                    runner.set_inputs(task_data)  # type: ignore
                    loop.run_until_complete(runner.run_pipeline())

                    # 同步并报告结果
                    worker.sync_and_report(
                        task_data["task_id"],
                        "success",
                        result_queue,
                        save_video_path=task_data["save_video_path"],
                        message="推理完成",
                    )
                except Exception as e:
                    logger.error(f"进程 {rank} 处理任务时发生错误: {str(e)}")

                    # 同步并报告错误
                    worker.sync_and_report(
                        task_data.get("task_id", "unknown"),
                        "failed",
                        result_queue,
                        error=str(e),
                        message=f"推理失败: {str(e)}",
                    )

    except KeyboardInterrupt:
        logger.info(f"进程 {rank} 收到 KeyboardInterrupt，优雅退出")
    except Exception as e:
        logger.error(f"分布式推理服务进程 {rank} 启动失败: {str(e)}")
        if rank == 0:
            error_result = {
                "task_id": "startup",
                "status": "startup_failed",
                "error": str(e),
                "message": f"推理服务启动失败: {str(e)}",
            }
            result_queue.put(error_result)
    finally:
        # 清理资源
        try:
            if loop and not loop.is_closed():
                loop.close()
        except:  # noqa: E722
            pass

        try:
            if worker:
                worker.cleanup()
        except:  # noqa: E722
            pass


class DistributedInferenceService:
    def __init__(self):
        self.task_queue = None
        self.result_queue = None
        self.processes = []
        self.is_running = False

    def start_distributed_inference(self, args) -> bool:
        if self.is_running:
            logger.warning("分布式推理服务已在运行")
            return True

        nproc_per_node = args.nproc_per_node
        if nproc_per_node <= 0:
            logger.error("nproc_per_node 必须大于0")
            return False

        try:
            import random

            master_addr = "127.0.0.1"
            master_port = str(random.randint(20000, 29999))
            logger.info(f"分布式推理服务 Master Addr: {master_addr}, Master Port: {master_port}")

            # 创建共享队列
            self.task_queue = mp.Queue()
            self.result_queue = mp.Queue()

            # 启动进程
            for rank in range(nproc_per_node):
                p = mp.Process(
                    target=_distributed_inference_worker,
                    args=(
                        rank,
                        nproc_per_node,
                        master_addr,
                        master_port,
                        args,
                        self.task_queue,
                        self.result_queue,
                    ),
                    daemon=True,
                )
                p.start()
                self.processes.append(p)

            self.is_running = True
            logger.info(f"分布式推理服务启动成功，共 {nproc_per_node} 个进程")
            return True

        except Exception as e:
            logger.exception(f"启动分布式推理服务时发生错误: {str(e)}")
            self.stop_distributed_inference()
            return False

    def stop_distributed_inference(self):
        if not self.is_running:
            return

        try:
            logger.info(f"正在停止 {len(self.processes)} 个分布式推理服务进程...")

            # 发送停止信号
            if self.task_queue:
                for _ in self.processes:
                    self.task_queue.put(None)

            # 等待进程结束
            for p in self.processes:
                try:
                    p.join(timeout=10)
                    if p.is_alive():
                        logger.warning(f"进程 {p.pid} 未在规定时间内结束，强制终止...")
                        p.terminate()
                        p.join(timeout=5)
                except:  # noqa: E722
                    pass

            logger.info("所有分布式推理服务进程已停止")

        except Exception as e:
            logger.error(f"停止分布式推理服务时发生错误: {str(e)}")
        finally:
            # 清理资源
            self._clean_queues()
            self.processes = []
            self.task_queue = None
            self.result_queue = None
            self.is_running = False

    def _clean_queues(self):
        for queue_obj in [self.task_queue, self.result_queue]:
            if queue_obj:
                try:
                    while not queue_obj.empty():
                        queue_obj.get_nowait()
                except:  # noqa: E722
                    pass

    def submit_task(self, task_data: dict) -> bool:
        if not self.is_running or not self.task_queue:
            logger.error("分布式推理服务未启动")
            return False

        try:
            self.task_queue.put(task_data)
            return True
        except Exception as e:
            logger.error(f"提交任务失败: {str(e)}")
            return False

    def wait_for_result(self, task_id: str, timeout: int = 300) -> Optional[dict]:
        if not self.is_running or not self.result_queue:
            return None

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                result = self.result_queue.get(timeout=1.0)

                if result.get("task_id") == task_id:
                    return result
                else:
                    # 不是当前任务的结果，放回队列
                    self.result_queue.put(result)
                    time.sleep(0.1)

            except queue.Empty:
                continue

        return None


class VideoGenerationService:
    def __init__(self, file_service: FileService, inference_service: DistributedInferenceService):
        self.file_service = file_service
        self.inference_service = inference_service

    async def generate_video(self, message: TaskRequest) -> TaskResponse:
        try:
            # 处理图片路径
            task_data = {
                "task_id": message.task_id,
                "prompt": message.prompt,
                "use_prompt_enhancer": message.use_prompt_enhancer,
                "negative_prompt": message.negative_prompt,
                "image_path": message.image_path,
                "num_fragments": message.num_fragments,
                "save_video_path": message.save_video_path,
                "infer_steps": message.infer_steps,
                "target_video_length": message.target_video_length,
                "seed": message.seed,
                "audio_path": message.audio_path,
                "video_duration": message.video_duration,
            }

            # 处理网络图片
            if message.image_path.startswith("http"):
                image_path = await self.file_service.download_image(message.image_path)
                task_data["image_path"] = str(image_path)

            # 处理输出路径
            save_video_path = self.file_service.get_output_path(message.save_video_path)
            task_data["save_video_path"] = str(save_video_path)

            # 提交任务到分布式推理服务
            if not self.inference_service.submit_task(task_data):
                raise RuntimeError("分布式推理服务未启动")

            # 等待结果
            result = self.inference_service.wait_for_result(message.task_id)

            if result is None:
                raise RuntimeError("任务处理超时")

            if result.get("status") == "success":
                ServiceStatus.complete_task(message)
                return TaskResponse(
                    task_id=message.task_id,
                    task_status="completed",
                    save_video_path=str(save_video_path),
                )
            else:
                error_msg = result.get("error", "推理失败")
                ServiceStatus.record_failed_task(message, error=error_msg)
                raise RuntimeError(error_msg)

        except Exception as e:
            logger.error(f"任务 {message.task_id} 处理失败: {str(e)}")
            ServiceStatus.record_failed_task(message, error=str(e))
            raise

    def get_task_result(self, task_id: str) -> TaskResultResponse:
        result = ServiceStatus.get_status_task_id(task_id)
        save_video_path = result.get("save_video_path")

        if save_video_path:
            file_path = Path(save_video_path)
            relative_path = file_path.relative_to(self.file_service.output_video_dir.resolve()) if str(file_path).startswith(str(self.file_service.output_video_dir.resolve())) else file_path.name

            return TaskResultResponse(
                status="success",
                task_status=result.get("status", "unknown"),
                filename=file_path.name,
                file_size=file_path.stat().st_size,
                download_url=f"/v1/file/download/{relative_path}",
                message="任务结果已准备就绪",
            )

        return TaskResultResponse(
            status="not_found",
            task_status=result.get("status", "unknown"),
            message="Task result not found",
        )
