import os
import gc
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import imageio
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.models.runners.wan.wan_runner import WanRunner
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.models.schedulers.wan.causvid.scheduler import WanCausVidScheduler
from lightx2v.models.schedulers.wan.causvid.df_scheduler import WanCausVidDfScheduler
from lightx2v.utils.profiler import ProfilingContext4Debug, ProfilingContext
from lightx2v.models.input_encoders.hf.t5.model import T5EncoderModel
from lightx2v.models.input_encoders.hf.xlm_roberta.model import CLIPModel
from lightx2v.models.networks.wan.causvid_model import WanCausVidModel
from lightx2v.models.networks.wan.lora_adapter import WanLoraWrapper
from lightx2v.models.video_encoders.hf.wan.vae import WanVAE
from lightx2v.utils.memory_profiler import peak_memory_decorator
from loguru import logger
import torch.distributed as dist
import pdb, os
import torchvision
import numpy as np
import imageio

def save_first_frame(videos: torch.Tensor, output_path: str):
    # 假设 videos 形状为 [B, C, T, H, W]
    if videos.dim() != 5:
        raise ValueError("videos tensor must have 5 dimensions [B, C, T, H, W]")

    first_frame = videos[0, :, 0, :, :]  # 取第一个视频的第0帧，形状[C, H, W]

    # 从 [-1, 1] 转成 [0, 1]
    first_frame = (first_frame + 1) / 2.0
    first_frame = first_frame.clamp(0, 1)

    # 转成 PIL.Image（RGB）
    image = to_pil_image(first_frame.cpu())  # 转为 CPU 并转成 PIL 图像

    # 保存图片
    image.save(output_path)


def save_last_frame(video_path, output_jpg_path):
    reader = imageio.get_reader(video_path)
    last_frame = reader.get_data(reader.count_frames() - 1)  # 读取最后一帧
    imageio.imwrite(output_jpg_path, last_frame)
    reader.close()

@RUNNER_REGISTER("wan2.1_causvid")
class WanCausVidRunner(WanRunner):
    def __init__(self, config):
        super().__init__(config)
        self.num_frame_per_block = self.model.config.num_frame_per_block
        self.frame_seq_length = self.model.config.frame_seq_length
        self.num_blocks = self.model.config.num_blocks
        self.block_overlap_frame =  self.model.config.block_overlap_frame

        self.init_block_iranges()

    def init_block_iranges(self):
        step   = self.num_frame_per_block - self.block_overlap_frame   # 每个 block 的移动步长
        s = np.arange(self.num_blocks - 1) * step                
        starts = s + self.num_frame_per_block    
        starts =  np.concatenate(([0], starts))

        ends = np.append(starts, starts[-1] + step)
        ends = ends[1:]
        self.block_ranges = list(zip(starts.tolist(), ends.tolist()))

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
        model = WanCausVidModel(self.config.model_path, self.config, init_device)

        if self.config.lora_path:
            lora_wrapper = WanLoraWrapper(model)
            lora_name = lora_wrapper.load_lora(self.config.lora_path)
            lora_wrapper.apply_lora(lora_name, self.config.strength_model)
            logger.info(f"Loaded LoRA: {lora_name}")

        vae_model = WanVAE(vae_pth=os.path.join(self.config.model_path, "Wan2.1_VAE.pth"), device=init_device, parallel=self.config.parallel_vae)
        if self.config.task == "i2v":
            image_encoder = CLIPModel(
                dtype=torch.float16,
                device=init_device,
                checkpoint_path=os.path.join(self.config.model_path, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
                tokenizer_path=os.path.join(self.config.model_path, "xlm-roberta-large"),
            )

        return model, text_encoders, vae_model, image_encoder

    def set_inputs(self, inputs):
        super().set_inputs(inputs)
        self.config["num_fragments"] = inputs.get("num_fragments", 1)
        self.num_fragments = self.config["num_fragments"]

    def init_scheduler(self):
        #scheduler = WanCausVidDfScheduler(self.config)
        #scheduler = WanCausVidDfScheduler(self.config)
        scheduler = WanCausVidScheduler(self.config)
        self.model.set_scheduler(scheduler)

    def set_target_shape(self):
        if self.config.task == "i2v":
            self.config.target_shape = (16, self.config.num_frame_per_block, self.config.lat_h, self.config.lat_w)
            # i2v需根据input shape重置frame_seq_length
            frame_seq_length = (self.config.lat_h // 2) * (self.config.lat_w // 2)
            self.model.transformer_infer.frame_seq_length = frame_seq_length
            self.frame_seq_length = frame_seq_length
        elif self.config.task == "t2v":
            self.config.target_shape = (
                16,
                self.config.num_frame_per_block,
                int(self.config.target_height) // self.config.vae_stride[1],
                int(self.config.target_width) // self.config.vae_stride[2],
            )

    @peak_memory_decorator
    def run(self):
        #self.model.transformer_infer._init_kv_cache(self.config.total_latens_frames_num, dtype=torch.bfloat16, device="cuda")
       # self.model.transformer_infer._init_crossattn_cache(dtype=torch.bfloat16, device="cuda")

        output_images = [] 
        #pdb.set_trace()
        #breakpoint()
        kv_start = 0
        kv_end = kv_start + self.num_frame_per_block * self.frame_seq_length

        debug_image = None
        for block_idx in range(self.num_blocks):

            s, e = self.block_ranges[block_idx] 
            kv_start = s * self.frame_seq_length
            kv_end   = e * self.frame_seq_length

            logger.info(f"=====> block_idx: {block_idx + 1} / {self.num_blocks}")
            logger.info(f"=====> kv_start: {kv_start}, kv_end: {kv_end}")

            if block_idx > 0:
                self.run_input_encoder(debug_image)

            for step_index in range(self.model.scheduler.infer_steps):
                logger.info(f"==> step_index: {step_index + 1} / {self.model.scheduler.infer_steps}")

                with ProfilingContext4Debug("step_pre"):
                    self.model.scheduler.step_pre(step_index, block_idx)

                with ProfilingContext4Debug("infer"):
                    self.model.infer(self.inputs,kv_start, kv_end)

                with ProfilingContext4Debug("step_post"):
                    self.model.scheduler.step_post()


            images = self.run_vae(self.model.scheduler.latents, self.model.scheduler.generator)
            logger.info(f"images: {images.shape}, latents:{self.model.scheduler.latents.shape}")
            value_range=(-1,1)
            images = images.clamp(min(value_range), max(value_range))
            images = torch.stack(
                [torchvision.utils.make_grid(u, nrow=1, normalize=True, value_range=value_range) for u in images.unbind(2)],
                dim=1,
            ).permute(1, 2, 3, 0)
            images = (images * 255).type(torch.uint8).cpu()
            logger.info(f"images: {images.shape}")

            pre_block_tail = images[-self.block_overlap_frame:]

            if pre_block_tail.dim() == 4:
                logger.info(f"pre_block_tail: {pre_block_tail.shape}")
                debug_image = torch.einsum("fhwc->fchw", pre_block_tail)
                logger.info(f"images: {debug_image.shape}")
                debug_image = to_pil_image(debug_image[0]) 
                debug_image.save(f"./{block_idx}.jpg")

            if block_idx == 0:
                output_images.append(images)
            else:
                output_images.append(images[self.block_overlap_frame:])

        writer = imageio.get_writer(self.config.save_video_path, fps=16, codec="libx264", quality=8)
        for block_frames in output_images:
            for frame in block_frames.numpy():
                writer.append_data(frame)
        writer.close()

    def end_run(self):
        self.model.scheduler.clear()
        del self.inputs, self.model.scheduler
        gc.collect()
        torch.cuda.empty_cache()

    def run_image_encoder(self, config, image_encoder, vae_model, img=None):
        if img is None:
            img = Image.open(config.image_path).convert("RGB")
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).cuda()
        logger.info(f"image_path: {config.image_path}, img:{img.shape}")
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

        logger.info(f"config.target_video_length:{config.target_video_length}")
        msk = torch.ones(1,25, lat_h, lat_w, device=torch.device("cuda"))
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        logger.info(f"msk:{msk.shape}")
        msk = msk.view(1, 7, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        padding = torch.concat(
                    [
                        torch.nn.functional.interpolate(img[None].cpu(), size=(h, w), mode="bicubic").transpose(0, 1),
                        torch.zeros(3, 27, h, w),
                    ],
                    dim=1,
                ).cuda()
        logger.info(f"vae_encodepadding_out:{padding.shape}, msk:{msk.shape}")
        vae_encode_out = vae_model.encode([padding], config)[0]
        logger.info(f"vae_encode_out:{vae_encode_out.shape}, msk:{msk.shape}")
        vae_encode_out = torch.concat([msk, vae_encode_out]).to(torch.bfloat16)

        return {"clip_encoder_out": clip_encoder_out, "vae_encode_out": vae_encode_out}
    

    def run_input_encoder(self, img=None):
        image_encoder_output = None
        if os.path.isfile(self.config.image_path):
            with ProfilingContext("Run Img Encoder"):
                image_encoder_output = self.run_image_encoder(self.config, self.image_encoder, self.vae_model, img=img)
        with ProfilingContext("Run Text Encoder"):
                text_encoder_output = self.run_text_encoder(self.config["prompt"], self.text_encoders, self.config, image_encoder_output)
        self.set_target_shape()
        self.inputs = {"text_encoder_output": text_encoder_output, "image_encoder_output": image_encoder_output}
        gc.collect()
        torch.cuda.empty_cache()


    def run_pipeline(self):
        if self.config["use_prompt_enhancer"]:
            self.config["prompt_enhanced"] = self.prompt_enhancer(self.config["prompt"])
        self.init_scheduler()
        self.run_input_encoder()
        self.model.scheduler.prepare(self.inputs["image_encoder_output"])
        self.run()
        self.end_run()
        gc.collect()
        torch.cuda.empty_cache()
