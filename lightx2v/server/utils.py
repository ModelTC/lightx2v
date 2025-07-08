import sys
import psutil
import signal
import base64
from PIL import Image
from loguru import logger
from typing import Optional
from datetime import datetime
from pydantic import BaseModel
import threading
import torch
import io


class ProcessManager:
    """进程管理器"""

    @staticmethod
    def kill_all_related_processes():
        """杀死当前进程及其所有子进程"""
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except Exception as e:
                logger.info(f"Failed to kill child process {child.pid}: {e}")
        try:
            current_process.kill()
        except Exception as e:
            logger.info(f"Failed to kill main process: {e}")

    @staticmethod
    def signal_handler(sig, frame):
        logger.info("\nReceived Ctrl+C, shutting down all related processes...")
        ProcessManager.kill_all_related_processes()
        sys.exit(0)

    @staticmethod
    def register_signal_handler():
        """注册SIGINT信号处理器"""
        signal.signal(signal.SIGINT, ProcessManager.signal_handler)


class TaskStatusMessage(BaseModel):
    """任务状态消息模型"""

    task_id: str


class ServiceStatus:
    """服务状态管理器"""

    _lock = threading.Lock()
    _current_task = None
    _result_store = {}

    @classmethod
    def start_task(cls, message):
        """开始任务"""
        with cls._lock:
            if cls._current_task is not None:
                raise RuntimeError("Service busy")
            if message.task_id_must_unique and message.task_id in cls._result_store:
                raise RuntimeError(f"Task ID {message.task_id} already exists")
            cls._current_task = {"message": message, "start_time": datetime.now()}
            return message.task_id

    @classmethod
    def complete_task(cls, message):
        """完成任务"""
        with cls._lock:
            if cls._current_task:
                cls._result_store[message.task_id] = {
                    "success": True,
                    "message": message,
                    "start_time": cls._current_task["start_time"],
                    "completion_time": datetime.now(),
                    "save_video_path": message.save_video_path,
                }
                cls._current_task = None

    @classmethod
    def record_failed_task(cls, message, error: Optional[str] = None):
        with cls._lock:
            if cls._current_task:
                cls._result_store[message.task_id] = {"success": False, "message": message, "start_time": cls._current_task["start_time"], "error": error, "save_video_path": message.save_video_path}
                cls._current_task = None

    @classmethod
    def clean_stopped_task(cls):
        with cls._lock:
            if cls._current_task:
                message = cls._current_task["message"]
                error = "Task stopped by user"
                cls._result_store[message.task_id] = {"success": False, "message": message, "start_time": cls._current_task["start_time"], "error": error, "save_video_path": message.save_video_path}
                cls._current_task = None

    @classmethod
    def get_status_task_id(cls, task_id: str):
        """根据任务ID获取状态"""
        with cls._lock:
            if cls._current_task and cls._current_task["message"].task_id == task_id:
                return {"status": "processing", "task_id": task_id}
            if task_id in cls._result_store:
                result = cls._result_store[task_id]
                return {
                    "status": "completed" if result["success"] else "failed",
                    "task_id": task_id,
                    "success": result["success"],
                    "start_time": result["start_time"],
                    "completion_time": result.get("completion_time"),
                    "error": result.get("error"),
                    "save_video_path": result.get("save_video_path"),
                }
            return {"status": "not_found", "task_id": task_id}

    @classmethod
    def get_status_service(cls):
        """获取服务状态"""
        with cls._lock:
            if cls._current_task:
                return {"service_status": "busy", "task_id": cls._current_task["message"].task_id, "start_time": cls._current_task["start_time"]}
            return {"service_status": "idle"}

    @classmethod
    def get_all_tasks(cls):
        """获取所有任务"""
        with cls._lock:
            return cls._result_store


class TensorTransporter:
    """张量传输器"""

    def __init__(self):
        self.buffer = io.BytesIO()

    def to_device(self, data, device):
        """将数据移动到指定设备"""
        if isinstance(data, dict):
            return {key: self.to_device(value, device) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.to_device(item, device) for item in data]
        elif isinstance(data, torch.Tensor):
            return data.to(device)
        else:
            return data

    def prepare_tensor(self, data) -> str:
        """准备张量数据"""
        self.buffer.seek(0)
        self.buffer.truncate()
        torch.save(self.to_device(data, "cpu"), self.buffer)
        return base64.b64encode(self.buffer.getvalue()).decode("utf-8")

    def load_tensor(self, tensor_base64: str, device="cuda"):
        """加载张量数据"""
        tensor_bytes = base64.b64decode(tensor_base64)
        with io.BytesIO(tensor_bytes) as buffer:
            return self.to_device(torch.load(buffer), device)


class ImageTransporter:
    """图像传输器"""

    def __init__(self):
        self.buffer = io.BytesIO()

    def prepare_image(self, image: Image.Image):
        """准备图像数据"""
        self.buffer.seek(0)
        self.buffer.truncate()
        image.save(self.buffer, format="PNG")
        return base64.b64encode(self.buffer.getvalue()).decode("utf-8")

    def load_image(self, image_base64: bytes) -> Image.Image:
        """加载图像数据"""
        image_bytes = base64.b64decode(image_base64)
        with io.BytesIO(image_bytes) as buffer:
            return Image.open(buffer).convert("RGB")
