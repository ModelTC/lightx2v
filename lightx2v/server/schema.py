from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from ..utils.generate_task_id import generate_task_id


class TaskRequest(BaseModel):
    task_id: str = Field(default_factory=generate_task_id, description="任务ID（自动生成）")
    prompt: str = Field("", description="生成提示词")
    use_prompt_enhancer: bool = Field(False, description="是否使用提示词增强")
    negative_prompt: str = Field("", description="负面提示词")
    image_path: str = Field("", description="输入图片路径")
    num_fragments: int = Field(1, description="片段数量")
    save_video_path: str = Field("", description="保存视频路径（可选，默认使用task_id.mp4）")
    infer_steps: int = Field(5, description="推理步数")
    target_video_length: int = Field(81, description="目标视频长度")
    seed: int = Field(42, description="随机种子")
    audio_path: str = Field("", description="输入音频路径（Wan-Audio）")
    video_duration: int = Field(5, description="视频时长（Wan-Audio）")

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
    prompt: str = Field(
        "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline'''s intricate details and the refreshing atmosphere of the seaside",
        description="生成提示词",
    )
    save_video_path: str = Field("", description="保存视频路径（可选，默认使用task_id.mp4）")
    use_prompt_enhancer: bool = False
    negative_prompt: str = "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    num_fragments: int = 1
    infer_steps: int = Field(5, description="推理步数")
    target_video_length: int = Field(81, description="目标视频长度， 默认81帧5秒（5 * 16 + 1）")
    seed: int = Field(42, description="随机种子")

    def __init__(self, **data):
        super().__init__(**data)
        # 如果save_video_path为空，则使用task_id.mp4
        if not self.save_video_path:
            self.save_video_path = f"{self.task_id}.mp4"
