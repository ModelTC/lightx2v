from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from ..utils.generate_task_id import generate_task_id


class TaskRequest(BaseModel):
    task_id: str = Field(default_factory=generate_task_id, description="任务ID（自动生成）")
    prompt: str = Field(..., description="生成提示词")
    use_prompt_enhancer: bool = Field(False, description="是否使用提示词增强")
    negative_prompt: str = Field("", description="负面提示词")
    image_path: str = Field("", description="输入图片路径")
    num_fragments: int = Field(1, description="片段数量")
    save_video_path: str = Field("", description="保存视频路径（可选，默认使用task_id.mp4）")

    def __init__(self, **data):
        super().__init__(**data)
        # 如果save_video_path为空，则使用task_id.mp4
        if not self.save_video_path:
            self.save_video_path = f"{self.task_id}.mp4"

    def get(self, key, default=None):
        return getattr(self, key, default)


class TaskStatusMessage(BaseModel):
    task_id: str = Field(..., description="任务ID")


class TaskResponse(BaseModel):
    task_id: str
    task_status: str
    save_video_path: str


class TaskResultResponse(BaseModel):
    status: str
    task_status: str
    filename: Optional[str] = None
    file_size: Optional[int] = None
    download_url: Optional[str] = None
    message: str


class ServiceStatusResponse(BaseModel):
    service_status: str
    task_id: Optional[str] = None
    start_time: Optional[datetime] = None


class StopTaskResponse(BaseModel):
    stop_status: str
    reason: str


class FileUploadRequest(BaseModel):
    task_id: str = Field(default_factory=generate_task_id, description="任务ID（自动生成）")
    prompt: str
    save_video_path: str = Field("", description="保存视频路径（可选，默认使用task_id.mp4）")
    use_prompt_enhancer: bool = False
    negative_prompt: str = ""
    num_fragments: int = 1

    def __init__(self, **data):
        super().__init__(**data)
        # 如果save_video_path为空，则使用task_id.mp4
        if not self.save_video_path:
            self.save_video_path = f"{self.task_id}.mp4"
