import argparse
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger
import uvicorn
import json
import os
import torch

from lightx2v.models.input_encoders.hf.t5.model import T5EncoderModel
from lightx2v.models.input_encoders.hf.llama.model import TextEncoderHFLlamaModel
from lightx2v.models.input_encoders.hf.clip.model import TextEncoderHFClipModel
from lightx2v.models.input_encoders.hf.llava.model import TextEncoderHFLlavaModel

from lightx2v.utils.profiler import ProfilingContext
from lightx2v.utils.set_config import set_config
from lightx2v.utils.memory_profiler import peak_memory_decorator
from lightx2v.utils.service_utils import TaskStatusMessage, BaseServiceStatus, ProcessManager, TensorTransporter, ImageTransporter

tensor_transporter = TensorTransporter()
image_transporter = ImageTransporter()

# =========================
# FastAPI Related Code
# =========================

runner = None

app = FastAPI()


class Message(BaseModel):
    task_id: str
    task_id_must_unique: bool = False

    text: str
    img: Optional[bytes] = None

    def get(self, key, default=None):
        return getattr(self, key, default)


class TextEncoderServiceStatus(BaseServiceStatus):
    pass


class TextEncoderRunner:
    def __init__(self, config):
        self.config = config
        self.text_encoders = self.get_text_encoder_model()

    def get_text_encoder_model(self):
        if "wan2.1" in self.config.model_cls:
            text_encoder = T5EncoderModel(
                text_len=self.config["text_len"],
                dtype=torch.bfloat16,
                device="cuda",
                checkpoint_path=os.path.join(self.config.model_path, "models_t5_umt5-xxl-enc-bf16.pth"),
                tokenizer_path=os.path.join(self.config.model_path, "google/umt5-xxl"),
                shard_fn=None,
            )
            text_encoders = [text_encoder]
        elif self.config.model_cls in ["hunyuan"]:
            if self.config.task == "t2v":
                text_encoder_1 = TextEncoderHFLlamaModel(os.path.join(self.config.model_path, "text_encoder"), "cuda")
            else:
                text_encoder_1 = TextEncoderHFLlavaModel(os.path.join(self.config.model_path, "text_encoder_i2v"), "cuda")
            text_encoder_2 = TextEncoderHFClipModel(os.path.join(self.config.model_path, "text_encoder_2"), "cuda")
            text_encoders = [text_encoder_1, text_encoder_2]
        else:
            raise ValueError(f"Unsupported model class: {self.config.model_cls}")
        return text_encoders

    @peak_memory_decorator
    def _run_text_encoder(self, text, img):
        if "wan2.1" in self.config.model_cls:
            text_encoder_output = {}
            n_prompt = self.config.get("negative_prompt", "")
            context = self.text_encoders[0].infer([text], self.config)
            context_null = self.text_encoders[0].infer([n_prompt if n_prompt else ""], self.config)
            text_encoder_output["context"] = context
            text_encoder_output["context_null"] = context_null
        elif self.config.model_cls in ["hunyuan"]:
            text_encoder_output = {}
            for i, encoder in enumerate(self.text_encoders):
                if self.config.task == "i2v" and i == 0:
                    img = image_transporter.load_image(img)
                    text_state, attention_mask = encoder.infer(text, img, self.config)
                else:
                    text_state, attention_mask = encoder.infer(text, self.config)
                text_encoder_output[f"text_encoder_{i + 1}_text_states"] = text_state.to(dtype=torch.bfloat16)
                text_encoder_output[f"text_encoder_{i + 1}_attention_mask"] = attention_mask
        else:
            raise ValueError(f"Unsupported model class: {self.config.model_cls}")
        return text_encoder_output


def run_text_encoder(message: Message):
    try:
        global runner
        text_encoder_output = runner._run_text_encoder(message.text, message.img)
        TextEncoderServiceStatus.complete_task(message)
        return text_encoder_output
    except Exception as e:
        logger.error(f"task_id {message.task_id} failed: {str(e)}")
        TextEncoderServiceStatus.record_failed_task(message, error=str(e))


@app.post("/v1/local/text_encoder/generate")
def v1_local_text_encoder_generate(message: Message):
    try:
        task_id = TextEncoderServiceStatus.start_task(message)
        text_encoder_output = run_text_encoder(message)
        output = tensor_transporter.prepare_tensor(text_encoder_output)
        del text_encoder_output
        return {"task_id": task_id, "task_status": "completed", "output": output, "kwargs": None}
    except RuntimeError as e:
        return {"error": str(e)}


@app.get("/v1/local/text_encoder/generate/service_status")
async def get_service_status():
    return TextEncoderServiceStatus.get_status_service()


@app.get("/v1/local/text_encoder/generate/get_all_tasks")
async def get_all_tasks():
    return TextEncoderServiceStatus.get_all_tasks()


@app.post("/v1/local/text_encoder/generate/task_status")
async def get_task_status(message: TaskStatusMessage):
    return TextEncoderServiceStatus.get_status_task_id(message.task_id)


# =========================
# Main Entry
# =========================

if __name__ == "__main__":
    ProcessManager.register_signal_handler()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cls", type=str, required=True, choices=["wan2.1", "hunyuan", "wan2.1_causvid", "wan2.1_skyreels_v2_df"], default="hunyuan")
    parser.add_argument("--task", type=str, choices=["t2v", "i2v"], default="t2v")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_json", type=str, required=True)

    parser.add_argument("--port", type=int, default=9002)
    args = parser.parse_args()
    logger.info(f"args: {args}")

    with ProfilingContext("Init Server Cost"):
        config = set_config(args)
        config["mode"] = "split_server"
        logger.info(f"config:\n{json.dumps(config, ensure_ascii=False, indent=4)}")
        runner = TextEncoderRunner(config)

    uvicorn.run(app, host="0.0.0.0", port=config.port, reload=False, workers=1)
