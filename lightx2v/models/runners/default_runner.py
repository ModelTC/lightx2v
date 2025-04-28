import os
import gc
import torch
import torch.distributed as dist
from lightx2v.utils.profiler import ProfilingContext4Debug, ProfilingContext
from lightx2v.utils.utils import save_videos_grid, cache_video
from lightx2v.utils.envs import *
from lightx2v.utils.prompt_enhancer import PromptEnhancer


class DefaultRunner:
    def __init__(self, config):
        self.config = config

        self.prompt_enhancer_cpu_offload = False
        self.gpu_count = torch.cuda.device_count()
        self.all_gpus = os.environ["CUDA_VISIBLE_DEVICES"]
        self.video_gpus = os.environ["CUDA_VISIBLE_DEVICES"]

        if self.config.prompt_enhancer is not None and self.config.task == "t2v":
            self.prompt_enhancer_cpu_offload = self.config.prompt_enhancer_cpu_offload
            self.load_prompt_enhancer()
            if not self.prompt_enhancer_cpu_offload:
                self.video_gpus = ",".join(self.all_gpus.split(",")[: self.gpu_count - 1])

        print(f"Prompt Enhancer CPU Offload: {self.prompt_enhancer_cpu_offload}")
        print(f"All GPUs: {self.all_gpus}")
        print(f"Video GPUs: {self.video_gpus}")

        os.environ["CUDA_VISIBLE_DEVICES"] = self.video_gpus
        self.model, self.text_encoders, self.vae_model, self.image_encoder = self.load_model()
        os.environ["CUDA_VISIBLE_DEVICES"] = self.all_gpus

    @ProfilingContext("Load prompt enhancer")
    def load_prompt_enhancer(self):
        gpu_count = torch.cuda.device_count()

        if gpu_count == 1:
            self.prompt_enhancer_cpu_offload = True

        self.prompt_enhancer = PromptEnhancer(model_name=self.config.prompt_enhancer, device_map="cpu" if self.prompt_enhancer_cpu_offload else f"cuda:{self.gpu_count - 1}")

    def run_prompt_enhancer(self):
        if self.config.prompt_enhancer is None or self.config.task != "t2v":
            return

        if self.prompt_enhancer_cpu_offload:
            self.prompt_enhancer.to_device(f"cuda:{self.gpu_count - 1}")

        enhanced_prompt = self.prompt_enhancer(self.config.prompt)
        print(f"Original prompt: {self.config.prompt}")
        print(f"Enhanced prompt: {enhanced_prompt}")
        self.config.prompt = enhanced_prompt

        if self.prompt_enhancer_cpu_offload:
            self.prompt_enhancer.to_device("cpu")
            if not self.config.cpu_offload:
                self.model_to_device("cuda")

    def model_to_device(self, device):
        if device == "cuda" and self.config["parallel_attn_type"]:
            cur_rank = dist.get_rank()
            torch.cuda.set_device(cur_rank)

        self.model, self.text_encoders, self.vae_model, self.image_encoder = self.model.to(device), self.text_encoders.to(device), self.vae_model.to(device), self.image_encoder.to(device)

    def set_inputs(self, inputs):
        self.config["prompt"] = inputs.get("prompt", "")
        self.config["negative_prompt"] = inputs.get("negative_prompt", "")
        self.config["image_path"] = inputs.get("image_path", "")
        self.config["save_video_path"] = inputs.get("save_video_path", "")

    def run_input_encoder(self):
        image_encoder_output = None
        if self.config["task"] == "i2v":
            with ProfilingContext("Run Img Encoder"):
                image_encoder_output = self.run_image_encoder(self.config, self.image_encoder, self.vae_model)
        with ProfilingContext("Run Text Encoder"):
            text_encoder_output = self.run_text_encoder(self.config["prompt"], self.text_encoders, self.config, image_encoder_output)
        self.set_target_shape()
        self.inputs = {"text_encoder_output": text_encoder_output, "image_encoder_output": image_encoder_output}

        gc.collect()
        torch.cuda.empty_cache()

    def run(self):
        for step_index in range(self.model.scheduler.infer_steps):
            print(f"==> step_index: {step_index + 1} / {self.model.scheduler.infer_steps}")

            with ProfilingContext4Debug("step_pre"):
                self.model.scheduler.step_pre(step_index=step_index)

            with ProfilingContext4Debug("infer"):
                self.model.infer(self.inputs)

            with ProfilingContext4Debug("step_post"):
                self.model.scheduler.step_post()

        return self.model.scheduler.latents, self.model.scheduler.generator

    def run_step(self, step_index=0):
        self.init_scheduler()
        self.run_input_encoder()
        self.model.scheduler.prepare(self.inputs["image_encoder_output"])
        self.model.scheduler.step_pre(step_index=step_index)
        self.model.infer(self.inputs)
        self.model.scheduler.step_post()

    def end_run(self):
        self.model.scheduler.clear()
        del self.inputs, self.model.scheduler
        torch.cuda.empty_cache()

    @ProfilingContext("Run VAE")
    def run_vae(self, latents, generator):
        images = self.vae_model.decode(latents, generator=generator, config=self.config)
        return images

    @ProfilingContext("Save video")
    def save_video(self, images):
        if not self.config.parallel_attn_type or (self.config.parallel_attn_type and dist.get_rank() == 0):
            if self.config.model_cls in ["wan2.1", "wan2.1_causal"]:
                cache_video(tensor=images, save_file=self.config.save_video_path, fps=16, nrow=1, normalize=True, value_range=(-1, 1))
            else:
                save_videos_grid(images, self.config.save_video_path, fps=24)

    def run_pipeline(self):
        self.run_prompt_enhancer()
        self.init_scheduler()
        self.run_input_encoder()
        self.model.scheduler.prepare(self.inputs["image_encoder_output"])
        latents, generator = self.run()
        self.end_run()
        images = self.run_vae(latents, generator)
        self.save_video(images)
        del latents, generator, images
        gc.collect()
        torch.cuda.empty_cache()
        if self.prompt_enhancer_cpu_offload:
            self.model_to_device("cpu")
