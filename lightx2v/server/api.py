import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
import threading
import gc
import torch
from pathlib import Path
import uuid

from .schema import TaskRequest, TaskResponse, TaskStatusMessage, TaskResultResponse, ServiceStatusResponse, StopTaskResponse
from .service import FileService, DistributedInferenceService, VideoGenerationService
from .utils import ServiceStatus


class ApiServer:
    """API服务器"""

    def __init__(self):
        self.app = FastAPI(title="LightX2V API", version="1.0.0")
        self.file_service = None
        self.inference_service = None
        self.video_service = None
        self.thread = None
        self.stop_generation_event = threading.Event()

        self._setup_routes()

    def _setup_routes(self):
        """设置路由"""

        @self.app.post("/v1/local/video/generate", response_model=TaskResponse)
        async def generate_video(message: TaskRequest):
            """生成视频"""
            try:
                task_id = ServiceStatus.start_task(message)

                # 使用后台线程处理长时间运行的任务
                self.stop_generation_event.clear()
                self.thread = threading.Thread(target=self._process_video_generation, args=(message, self.stop_generation_event), daemon=True)
                self.thread.start()

                return TaskResponse(task_id=task_id, task_status="processing", save_video_path=message.save_video_path)
            except RuntimeError as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.post("/v1/local/video/generate_form", response_model=TaskResponse)
        async def generate_video_form(
            task_id: str,
            prompt: str,
            save_video_path: str,
            task_id_must_unique: bool = False,
            use_prompt_enhancer: bool = False,
            negative_prompt: str = "",
            num_fragments: int = 1,
            image_file: UploadFile = File(None),
        ):
            """通过表单生成视频"""
            # 处理上传的图片文件
            image_path = ""
            assert self.file_service is not None, "File service is not initialized"

            if image_file and image_file.filename:
                file_extension = Path(image_file.filename).suffix
                unique_filename = f"{uuid.uuid4()}{file_extension}"
                image_path = self.file_service.input_image_dir / unique_filename

                with open(image_path, "wb") as buffer:
                    content = await image_file.read()
                    buffer.write(content)

                image_path = str(image_path)

            message = TaskRequest(
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
                task_id = ServiceStatus.start_task(message)
                self.stop_generation_event.clear()
                self.thread = threading.Thread(target=self._process_video_generation, args=(message, self.stop_generation_event), daemon=True)
                self.thread.start()

                return TaskResponse(task_id=task_id, task_status="processing", save_video_path=message.save_video_path)
            except RuntimeError as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.get("/v1/local/video/generate/service_status", response_model=ServiceStatusResponse)
        async def get_service_status():
            """获取服务状态"""
            return ServiceStatus.get_status_service()

        @self.app.get("/v1/local/video/generate/get_all_tasks")
        async def get_all_tasks():
            """获取所有任务"""
            return ServiceStatus.get_all_tasks()

        @self.app.post("/v1/local/video/generate/task_status")
        async def get_task_status(message: TaskStatusMessage):
            """获取任务状态"""
            return ServiceStatus.get_status_task_id(message.task_id)

        @self.app.get("/v1/local/video/generate/get_task_result", response_model=TaskResultResponse)
        async def get_task_result(message: TaskStatusMessage):
            """获取任务结果"""
            assert self.video_service is not None, "Video service is not initialized"
            return self.video_service.get_task_result(message.task_id)

        @self.app.get("/v1/file/download/{file_path:path}")
        async def download_file(file_path: str):
            """下载文件"""
            assert self.file_service is not None, "File service is not initialized"
            try:
                full_path = self.file_service.output_video_dir / file_path
                resolved_path = full_path.resolve()

                # 安全检查：确保文件在允许的目录内
                if not str(resolved_path).startswith(str(self.file_service.output_video_dir.resolve())):
                    raise HTTPException(status_code=403, detail="不允许访问该文件")

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

                    def file_stream_generator(file_path: str, chunk_size: int = 1024 * 1024):
                        with open(file_path, "rb") as file:
                            while chunk := file.read(chunk_size):
                                yield chunk

                    return StreamingResponse(file_stream_generator(str(resolved_path)), media_type=mime_type, headers=headers)
                else:
                    raise HTTPException(status_code=404, detail=f"文件未找到: {file_path}")
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"处理文件下载请求时发生错误: {e}")
                raise HTTPException(status_code=500, detail="文件下载失败")

        @self.app.get("/v1/local/video/generate/stop_running_task", response_model=StopTaskResponse)
        async def stop_running_task():
            """停止运行中的任务"""
            if self.thread and self.thread.is_alive():
                try:
                    logger.info("正在发送停止信号给运行中的任务线程...")
                    self.stop_generation_event.set()
                    self.thread.join(timeout=5)

                    if self.thread.is_alive():
                        logger.warning("任务线程未在规定时间内停止，可能需要手动干预。")
                        return StopTaskResponse(stop_status="warning", reason="任务线程未在规定时间内停止，可能需要手动干预。")
                    else:
                        self.thread = None
                        ServiceStatus.clean_stopped_task()
                        gc.collect()
                        torch.cuda.empty_cache()
                        logger.info("任务已成功停止。")
                        return StopTaskResponse(stop_status="success", reason="Task stopped successfully.")
                except Exception as e:
                    logger.error(f"停止任务时发生错误: {str(e)}")
                    return StopTaskResponse(stop_status="error", reason=str(e))
            else:
                return StopTaskResponse(stop_status="do_nothing", reason="No running task found.")

    def _process_video_generation(self, message: TaskRequest, stop_event: threading.Event):
        """处理视频生成（在后台线程中运行）"""
        assert self.video_service is not None, "Video service is not initialized"
        try:
            if stop_event.is_set():
                logger.info(f"任务 {message.task_id} 收到停止信号，正在终止")
                ServiceStatus.record_failed_task(message, error="任务被停止")
                return

            # 使用视频生成服务处理任务
            result = asyncio.run(self.video_service.generate_video(message))

        except Exception as e:
            logger.error(f"任务 {message.task_id} 处理失败: {str(e)}")
            ServiceStatus.record_failed_task(message, error=str(e))

    def initialize_services(self, cache_dir: Path, inference_service: DistributedInferenceService):
        """初始化服务"""
        self.file_service = FileService(cache_dir)
        self.inference_service = inference_service
        self.video_service = VideoGenerationService(self.file_service, inference_service)

    def get_app(self) -> FastAPI:
        """获取FastAPI应用"""
        return self.app
