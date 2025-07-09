import asyncio
from fastapi import FastAPI, UploadFile, HTTPException, Form, File, APIRouter
from fastapi.responses import StreamingResponse
from loguru import logger
import threading
import gc
import torch
from pathlib import Path
import uuid
from typing import Optional
from .schema import (
    TaskRequest,
    TaskResponse,
    ServiceStatusResponse,
    StopTaskResponse,
)
from .service import FileService, DistributedInferenceService, VideoGenerationService
from .utils import ServiceStatus


class ApiServer:
    def __init__(self):
        self.app = FastAPI(title="LightX2V API", version="1.0.0")
        self.file_service = None
        self.inference_service = None
        self.video_service = None
        self.thread = None
        self.stop_generation_event = threading.Event()

        # 创建路由器
        self.tasks_router = APIRouter(prefix="/v1/tasks", tags=["tasks"])
        self.files_router = APIRouter(prefix="/v1/files", tags=["files"])
        self.service_router = APIRouter(prefix="/v1/service", tags=["service"])

        self._setup_routes()

    def _setup_routes(self):
        """设置路由"""
        self._setup_task_routes()
        self._setup_file_routes()
        self._setup_service_routes()

        # 注册路由器
        self.app.include_router(self.tasks_router)
        self.app.include_router(self.files_router)
        self.app.include_router(self.service_router)

    def _stream_file_response(self, file_path: Path, filename: str | None = None) -> StreamingResponse:
        """公共的文件流响应方法"""
        assert self.file_service is not None, "File service is not initialized"

        try:
            resolved_path = file_path.resolve()

            # 安全检查：确保文件在允许的目录内
            if not str(resolved_path).startswith(str(self.file_service.output_video_dir.resolve())):
                raise HTTPException(status_code=403, detail="不允许访问该文件")

            if not resolved_path.exists() or not resolved_path.is_file():
                raise HTTPException(status_code=404, detail=f"文件未找到: {file_path}")

            file_size = resolved_path.stat().st_size
            actual_filename = filename or resolved_path.name

            # 设置适当的 MIME 类型
            mime_type = "application/octet-stream"
            if actual_filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                mime_type = "video/mp4"
            elif actual_filename.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
                mime_type = "image/jpeg"

            headers = {
                "Content-Disposition": f'attachment; filename="{actual_filename}"',
                "Content-Length": str(file_size),
                "Accept-Ranges": "bytes",
            }

            def file_stream_generator(file_path: str, chunk_size: int = 1024 * 1024):
                with open(file_path, "rb") as file:
                    while chunk := file.read(chunk_size):
                        yield chunk

            return StreamingResponse(
                file_stream_generator(str(resolved_path)),
                media_type=mime_type,
                headers=headers,
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"处理文件流响应时发生错误: {e}")
            raise HTTPException(status_code=500, detail="文件传输失败")

    def _setup_task_routes(self):
        @self.tasks_router.post("/", response_model=TaskResponse)
        async def create_task(message: TaskRequest):
            """创建视频生成任务"""
            try:
                task_id = ServiceStatus.start_task(message)

                # 使用后台线程处理长时间运行的任务
                self.stop_generation_event.clear()
                self.thread = threading.Thread(
                    target=self._process_video_generation,
                    args=(message, self.stop_generation_event),
                    daemon=True,
                )
                self.thread.start()

                return TaskResponse(
                    task_id=task_id,
                    task_status="processing",
                    save_video_path=message.save_video_path,
                )
            except RuntimeError as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.tasks_router.post("/form", response_model=TaskResponse)
        async def create_task_form(
            image_file: UploadFile = File(...),
            prompt: str = Form(default=""),
            save_video_path: str = Form(default=""),
            use_prompt_enhancer: bool = Form(default=False),
            negative_prompt: str = Form(default=""),
            num_fragments: int = Form(default=1),
            infer_steps: int = Form(default=5),
            target_video_length: int = Form(default=81),
            seed: int = Form(default=42),
            audio_file: Optional[UploadFile] = File(default=None),
            video_duration: int = Form(default=5),
        ):
            """通过表单创建视频生成任务"""
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

            audio_path = ""
            if audio_file and audio_file.filename:
                file_extension = Path(audio_file.filename).suffix
                unique_filename = f"{uuid.uuid4()}{file_extension}"
                audio_path = self.file_service.input_audio_dir / unique_filename

                with open(audio_path, "wb") as buffer:
                    content = await audio_file.read()
                    buffer.write(content)

                audio_path = str(audio_path)

            message = TaskRequest(
                prompt=prompt,
                use_prompt_enhancer=use_prompt_enhancer,
                negative_prompt=negative_prompt,
                image_path=image_path,
                num_fragments=num_fragments,
                save_video_path=save_video_path,
                infer_steps=infer_steps,
                target_video_length=target_video_length,
                seed=seed,
                audio_path=audio_path,
                video_duration=video_duration,
            )

            try:
                task_id = ServiceStatus.start_task(message)
                self.stop_generation_event.clear()
                self.thread = threading.Thread(
                    target=self._process_video_generation,
                    args=(message, self.stop_generation_event),
                    daemon=True,
                )
                self.thread.start()

                return TaskResponse(
                    task_id=task_id,
                    task_status="processing",
                    save_video_path=message.save_video_path,
                )
            except RuntimeError as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.tasks_router.get("/", response_model=list)
        async def list_tasks():
            """获取所有任务列表"""
            return ServiceStatus.get_all_tasks()

        @self.tasks_router.get("/{task_id}/status")
        async def get_task_status(task_id: str):
            """获取指定任务的状态"""
            return ServiceStatus.get_status_task_id(task_id)

        @self.tasks_router.get("/{task_id}/result")
        async def get_task_result(task_id: str):
            """获取指定任务的结果视频文件"""
            assert self.video_service is not None, "Video service is not initialized"
            assert self.file_service is not None, "File service is not initialized"

            try:
                task_status = ServiceStatus.get_status_task_id(task_id)

                if not task_status or task_status.get("status") != "completed":
                    raise HTTPException(status_code=404, detail="任务未完成或不存在")

                save_video_path = task_status.get("save_video_path")
                if not save_video_path:
                    raise HTTPException(status_code=404, detail="任务结果文件不存在")

                full_path = Path(save_video_path)
                if not full_path.is_absolute():
                    full_path = self.file_service.output_video_dir / save_video_path

                return self._stream_file_response(full_path)

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"获取任务结果时发生错误: {e}")
                raise HTTPException(status_code=500, detail="获取任务结果失败")

        @self.tasks_router.delete("/running", response_model=StopTaskResponse)
        async def stop_running_task():
            """停止当前运行的任务"""
            if self.thread and self.thread.is_alive():
                try:
                    logger.info("正在发送停止信号给运行中的任务线程...")
                    self.stop_generation_event.set()
                    self.thread.join(timeout=5)

                    if self.thread.is_alive():
                        logger.warning("任务线程未在规定时间内停止，可能需要手动干预。")
                        return StopTaskResponse(
                            stop_status="warning",
                            reason="任务线程未在规定时间内停止，可能需要手动干预。",
                        )
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

    def _setup_file_routes(self):
        @self.files_router.get("/download/{file_path:path}")
        async def download_file(file_path: str):
            """下载文件"""
            assert self.file_service is not None, "File service is not initialized"

            try:
                full_path = self.file_service.output_video_dir / file_path
                return self._stream_file_response(full_path)
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"处理文件下载请求时发生错误: {e}")
                raise HTTPException(status_code=500, detail="文件下载失败")

    def _setup_service_routes(self):
        @self.service_router.get("/status", response_model=ServiceStatusResponse)
        async def get_service_status():
            """获取服务状态"""
            return ServiceStatus.get_status_service()

    def _process_video_generation(self, message: TaskRequest, stop_event: threading.Event):
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
        self.file_service = FileService(cache_dir)
        self.inference_service = inference_service
        self.video_service = VideoGenerationService(self.file_service, inference_service)

    def get_app(self) -> FastAPI:
        return self.app
