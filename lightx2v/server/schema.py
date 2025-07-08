from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class TaskRequest(BaseModel):
    """任务请求模型"""

    task_id: str = Field(..., description="任务ID")
    task_id_must_unique: bool = Field(False, description="任务ID是否必须唯一")
    prompt: str = Field(..., description="生成提示词")
    use_prompt_enhancer: bool = Field(False, description="是否使用提示词增强")
    negative_prompt: str = Field("", description="负面提示词")
    image_path: str = Field("", description="输入图片路径")
    num_fragments: int = Field(1, description="片段数量")
    save_video_path: str = Field(..., description="保存视频路径")

    def get(self, key, default=None):
        return getattr(self, key, default)


class TaskStatusMessage(BaseModel):
    """任务状态查询模型"""

    task_id: str = Field(..., description="任务ID")


class TaskResponse(BaseModel):
    """任务响应模型"""

    task_id: str
    task_status: str
    save_video_path: str


class TaskResultResponse(BaseModel):
    """任务结果响应模型"""

    status: str
    task_status: str
    filename: Optional[str] = None
    file_size: Optional[int] = None
    download_url: Optional[str] = None
    message: str


class ServiceStatusResponse(BaseModel):
    """服务状态响应模型"""

    service_status: str
    task_id: Optional[str] = None
    start_time: Optional[datetime] = None


class StopTaskResponse(BaseModel):
    """停止任务响应模型"""

    stop_status: str
    reason: str


class FileUploadRequest(BaseModel):
    """文件上传请求模型"""

    task_id: str
    prompt: str
    save_video_path: str
    task_id_must_unique: bool = False
    use_prompt_enhancer: bool = False
    negative_prompt: str = ""
    num_fragments: int = 1
