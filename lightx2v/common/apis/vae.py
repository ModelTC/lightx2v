import argparse
from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger
from typing import Optional
import numpy as np
import uvicorn
import json
import os
import torch
import torchvision
import torchvision.transforms.functional as TF

from lightx2v.models.video_encoders.hf.wan.vae import WanVAE
from lightx2v.models.video_encoders.hf.autoencoder_kl_causal_3d.model import VideoEncoderKLCausal3DModel
from lightx2v.models.runners.hunyuan.hunyuan_runner import HunyuanRunner

from lightx2v.utils.profiler import ProfilingContext
from lightx2v.utils.set_config import set_config
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

    img: Optional[bytes] = None
    latents: Optional[bytes] = None

    def get(self, key, default=None):
        return getattr(self, key, default)


class VAEServiceStatus(BaseServiceStatus):
    pass


class VAEEncoderRunner:
    def __init__(self, config):
        self.config = config
        self.vae_model = self.get_vae_model()

    def get_vae_model(self):
        if "wan2.1" in self.config.model_cls:
            vae_model = WanVAE(
                vae_pth=os.path.join(self.config.model_path, "Wan2.1_VAE.pth"),
                device="cuda",
                parallel=self.config.parallel_vae,
            )
        elif self.config.model_cls in ["hunyuan"]:
            vae_model = VideoEncoderKLCausal3DModel(model_path=self.config.model_path, dtype=torch.float16, device="cuda", config=self.config)
        else:
            raise ValueError(f"Unsupported model class: {self.config.model_cls}")
        return vae_model

    def _run_vae_encoder(self, img):
        img = image_transporter.load_image(img)
        kwargs = {}
        if "wan2.1" in self.config.model_cls:
            img = TF.to_tensor(img).sub_(0.5).div_(0.5).cuda()
            h, w = img.shape[1:]
            aspect_ratio = h / w
            max_area = self.config.target_height * self.config.target_width
            lat_h = round(np.sqrt(max_area * aspect_ratio) // self.config.vae_stride[1] // self.config.patch_size[1] * self.config.patch_size[1])
            lat_w = round(np.sqrt(max_area / aspect_ratio) // self.config.vae_stride[2] // self.config.patch_size[2] * self.config.patch_size[2])
            h = lat_h * self.config.vae_stride[1]
            w = lat_w * self.config.vae_stride[2]

            self.config.lat_h, kwargs["lat_h"] = lat_h, lat_h
            self.config.lat_w, kwargs["lat_w"] = lat_w, lat_w

            msk = torch.ones(1, self.config.target_video_length, lat_h, lat_w, device=torch.device("cuda"))
            msk[:, 1:] = 0
            msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
            msk = msk.transpose(1, 2)[0]
            vae_encode_out = self.vae_model.encode(
                [
                    torch.concat(
                        [
                            torch.nn.functional.interpolate(img[None].cpu(), size=(h, w), mode="bicubic").transpose(0, 1),
                            torch.zeros(3, self.config.target_video_length - 1, h, w),
                        ],
                        dim=1,
                    ).cuda()
                ],
                self.config,
            )[0]
            vae_encode_out = torch.concat([msk, vae_encode_out]).to(torch.bfloat16)
        elif self.config.model_cls in ["hunyuan"]:
            if self.config.i2v_resolution == "720p":
                bucket_hw_base_size = 960
            elif self.config.i2v_resolution == "540p":
                bucket_hw_base_size = 720
            elif self.config.i2v_resolution == "360p":
                bucket_hw_base_size = 480
            else:
                raise ValueError(f"self.config.i2v_resolution: {self.config.i2v_resolution} must be in [360p, 540p, 720p]")

            origin_size = img.size

            crop_size_list = HunyuanRunner.generate_crop_size_list(bucket_hw_base_size, 32)
            aspect_ratios = np.array([round(float(h) / float(w), 5) for h, w in crop_size_list])
            closest_size, closest_ratio = HunyuanRunner.get_closest_ratio(origin_size[1], origin_size[0], aspect_ratios, crop_size_list)

            self.config.target_height, self.config.target_width = closest_size
            kwargs["target_height"], kwargs["target_width"] = closest_size

            resize_param = min(closest_size)
            center_crop_param = closest_size

            ref_image_transform = torchvision.transforms.Compose(
                [torchvision.transforms.Resize(resize_param), torchvision.transforms.CenterCrop(center_crop_param), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.5], [0.5])]
            )

            semantic_image_pixel_values = [ref_image_transform(img)]
            semantic_image_pixel_values = torch.cat(semantic_image_pixel_values).unsqueeze(0).unsqueeze(2).to(torch.float16).to(torch.device("cuda"))

            vae_encode_out = self.vae_model.encode(semantic_image_pixel_values, self.config).mode()

            scaling_factor = 0.476986
            vae_encode_out.mul_(scaling_factor)
        else:
            raise ValueError(f"Unsupported model class: {self.config.model_cls}")
        return vae_encode_out, kwargs

    def _run_vae_decoder(self, latents):
        latents = tensor_transporter.load_tensor(latents)
        images = self.vae_model.decode(latents, generator=None, config=self.config)
        return images


def run_vae_encoder(message: Message):
    try:
        global runner
        vae_encode_out, kwargs = runner._run_vae_encoder(message.img)
        VAEServiceStatus.complete_task(message)
        return vae_encode_out, kwargs
    except Exception as e:
        logger.error(f"task_id {message.task_id} failed: {str(e)}")
        VAEServiceStatus.record_failed_task(message, error=str(e))


def run_vae_decoder(message: Message):
    try:
        global runner
        images = runner._run_vae_decoder(message.latents)
        VAEServiceStatus.complete_task(message)
        return images
    except Exception as e:
        logger.error(f"task_id {message.task_id} failed: {str(e)}")
        VAEServiceStatus.record_failed_task(message, error=str(e))


@app.post("/v1/local/vae_model/encoder/generate")
def v1_local_vae_model_encoder_generate(message: Message):
    try:
        task_id = VAEServiceStatus.start_task(message)
        vae_encode_out, kwargs = run_vae_encoder(message)
        output = tensor_transporter.prepare_tensor(vae_encode_out)
        del vae_encode_out
        return {"task_id": task_id, "task_status": "completed", "output": output, "kwargs": kwargs}
    except RuntimeError as e:
        return {"error": str(e)}


@app.post("/v1/local/vae_model/decoder/generate")
def v1_local_vae_model_decoder_generate(message: Message):
    try:
        task_id = VAEServiceStatus.start_task(message)
        vae_decode_out = run_vae_decoder(message)
        output = tensor_transporter.prepare_tensor(vae_decode_out)
        del vae_decode_out
        return {"task_id": task_id, "task_status": "completed", "output": output, "kwargs": None}
    except RuntimeError as e:
        return {"error": str(e)}


@app.get("/v1/local/vae_model/generate/service_status")
async def get_service_status():
    return VAEServiceStatus.get_status_service()


@app.get("/v1/local/vae_model/encoder/generate/service_status")
async def get_service_status():
    return VAEServiceStatus.get_status_service()


@app.get("/v1/local/vae_model/decoder/generate/service_status")
async def get_service_status():
    return VAEServiceStatus.get_status_service()


@app.get("/v1/local/vae_model/encoder/generate/get_all_tasks")
async def get_all_tasks():
    return VAEServiceStatus.get_all_tasks()


@app.get("/v1/local/vae_model/decoder/generate/get_all_tasks")
async def get_all_tasks():
    return VAEServiceStatus.get_all_tasks()


@app.post("/v1/local/vae_model/encoder/generate/task_status")
async def get_task_status(message: TaskStatusMessage):
    return VAEServiceStatus.get_status_task_id(message.task_id)


@app.post("/v1/local/vae_model/decoder/generate/task_status")
async def get_task_status(message: TaskStatusMessage):
    return VAEServiceStatus.get_status_task_id(message.task_id)


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

    parser.add_argument("--port", type=int, default=9004)
    args = parser.parse_args()
    logger.info(f"args: {args}")

    with ProfilingContext("Init Server Cost"):
        config = set_config(args)
        config["mode"] = "split_server"
        logger.info(f"config:\n{json.dumps(config, ensure_ascii=False, indent=4)}")
        runner = VAEEncoderRunner(config)

    uvicorn.run(app, host="0.0.0.0", port=config.port, reload=False, workers=1)
