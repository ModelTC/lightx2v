import os
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.models.schedulers.wan.feature_caching.scheduler import (
    WanSchedulerTeaCaching,
)
from lightx2v.utils.profiler import ProfilingContext
from lightx2v.models.input_encoders.hf.t5.model import T5EncoderModel
from lightx2v.models.input_encoders.hf.xlm_roberta.model import CLIPModel
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.networks.wan.lora_adapter import WanLoraWrapper
from lightx2v.models.video_encoders.hf.wan.vae import WanVAE
import torch.distributed as dist
from lightx2v.utils.memory_profiler import peak_memory_decorator
from loguru import logger


@RUNNER_REGISTER("wan2.1")
class WanRunner(DefaultRunner):
    def __init__(self, config):
        super().__init__(config)

    @ProfilingContext("Load models")
    def load_model(self):
        if self.config["parallel_attn_type"]:
            cur_rank = dist.get_rank()
            torch.cuda.set_device(cur_rank)
        image_encoder = None
        if self.config.cpu_offload:
            init_device = torch.device("cpu")
        else:
            init_device = torch.device("cuda")

        text_encoder = T5EncoderModel(
            text_len=self.config["text_len"],
            dtype=torch.bfloat16,
            device=init_device,
            checkpoint_path=os.path.join(self.config.model_path, "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=os.path.join(self.config.model_path, "google/umt5-xxl"),
            shard_fn=None,
        )
        text_encoders = [text_encoder]
        model = WanModel(self.config.model_path, self.config, init_device)

        if self.config.lora_path:
            lora_wrapper = WanLoraWrapper(model)
            lora_name = lora_wrapper.load_lora(self.config.lora_path)
            lora_wrapper.apply_lora(lora_name, self.config.strength_model)
            logger.info(f"Loaded LoRA: {lora_name}")

        vae_model = WanVAE(
            vae_pth=os.path.join(self.config.model_path, "Wan2.1_VAE.pth"),
            device=init_device,
            parallel=self.config.parallel_vae,
        )
        if self.config.task == "i2v":
            image_encoder = CLIPModel(
                dtype=torch.float16,
                device=init_device,
                checkpoint_path=os.path.join(
                    self.config.model_path,
                    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                ),
                tokenizer_path=os.path.join(self.config.model_path, "xlm-roberta-large"),
            )

        return model, text_encoders, vae_model, image_encoder

    def init_scheduler(self):
        if self.config.feature_caching == "NoCaching":
            scheduler = WanScheduler(self.config)
        elif self.config.feature_caching == "Tea":
            scheduler = WanSchedulerTeaCaching(self.config)
        else:
            raise NotImplementedError(f"Unsupported feature_caching type: {self.config.feature_caching}")
        self.model.set_scheduler(scheduler)

    @peak_memory_decorator
    def run_text_encoder(self, text, text_encoders, config, image_encoder_output):
        text_encoder_output = {}
        n_prompt = config.get("negative_prompt", "")
        context = text_encoders[0].infer([text], config)
        context_null = text_encoders[0].infer([n_prompt if n_prompt else ""], config)
        text_encoder_output["context"] = context
        text_encoder_output["context_null"] = context_null

        '''
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # 转换为GB
        logger.info(f"Peak Memory: {peak_memory:.2f} GB")
        del text_encoders[0]
        torch.cuda.empty_cache()  # 释放未使用但被缓存的显存
        torch.cuda.ipc_collect()  # 可选，强制释放跨进程缓存

        torch.cuda.reset_peak_memory_stats()
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # 转换为GB
        logger.info(f"Peak Memory: {peak_memory:.2f} GB")
        '''
        return text_encoder_output

    @peak_memory_decorator
    def run_image_encoder(self, config, image_encoder, vae_model):
        img = Image.open(config.image_path).convert("RGB")
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).cuda()
        clip_encoder_out = image_encoder.visual([img[:, None, :, :]], config).squeeze(0).to(torch.bfloat16)
        h, w = img.shape[1:]
        aspect_ratio = h / w
        max_area = config.target_height * config.target_width
        lat_h = round(np.sqrt(max_area * aspect_ratio) // config.vae_stride[1] // config.patch_size[1] * config.patch_size[1])
        lat_w = round(np.sqrt(max_area / aspect_ratio) // config.vae_stride[2] // config.patch_size[2] * config.patch_size[2])
        h = lat_h * config.vae_stride[1]
        w = lat_w * config.vae_stride[2]

        config.lat_h = lat_h
        config.lat_w = lat_w

        msk = torch.ones(1, config.target_video_length, lat_h, lat_w, device=torch.device("cuda"))
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]
        vae_encode_out = vae_model.encode(
            [
                torch.concat(
                    [
                        torch.nn.functional.interpolate(img[None].cpu(), size=(h, w), mode="bicubic").transpose(0, 1),
                        torch.zeros(3, config.target_video_length - 1, h, w),
                    ],
                    dim=1,
                ).cuda()
            ],
            config,
        )[0]
        vae_encode_out = torch.concat([msk, vae_encode_out]).to(torch.bfloat16)


        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # 转换为GB
        logger.info(f"Peak Memory: {peak_memory:.2f} GB")
        del image_encoder
        torch.cuda.empty_cache()  # 释放未使用但被缓存的显存
        torch.cuda.ipc_collect()  # 可选，强制释放跨进程缓存

        torch.cuda.reset_peak_memory_stats()
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # 转换为GB
        logger.info(f"Peak Memory: {peak_memory:.2f} GB")

        return {"clip_encoder_out": clip_encoder_out, "vae_encode_out": vae_encode_out}

    def set_target_shape(self):
        num_channels_latents = self.config.get("num_channels_latents", 16)
        if self.config.task == "i2v":
            self.config.target_shape = (
                num_channels_latents,
                (self.config.target_video_length - 1) // 4 + 1,
                self.config.lat_h,
                self.config.lat_w,
            )
        elif self.config.task == "t2v":
            self.config.target_shape = (
                num_channels_latents,
                (self.config.target_video_length - 1) // 4 + 1,
                int(self.config.target_height) // self.config.vae_stride[1],
                int(self.config.target_width) // self.config.vae_stride[2],
            )
