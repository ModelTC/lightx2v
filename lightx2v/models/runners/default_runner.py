import gc
import torch
import torch.distributed as dist
from lightx2v.utils.profiler import ProfilingContext4Debug, ProfilingContext
from lightx2v.utils.utils import save_videos_grid, cache_video
from lightx2v.utils.prompt_enhancer import PromptEnhancer
from lightx2v.utils.envs import *
from lightx2v.utils.memory_profiler import peak_memory_decorator
from loguru import logger


class DefaultRunner:
    def __init__(self, config):
        self.config = config
        if self.config.prompt_enhancer is not None and self.config.task == "t2v":
            self.load_prompt_enhancer()
        self.model, self.text_encoders, self.vae_model, self.image_encoder = self.load_model()

    @ProfilingContext("Load prompt enhancer")
    def load_prompt_enhancer(self):
        gpu_count = torch.cuda.device_count()
        if gpu_count == 1:
            logger.info("Only one GPU, use prompt enhancer cpu offload")
            raise NotImplementedError("prompt enhancer cpu offload is not supported.")
        self.prompt_enhancer = PromptEnhancer(model_name=self.config.prompt_enhancer, device_map="cuda:1")
        self.config["use_prompt_enhancer"] = True  # Set use_prompt_enhancer to True now. (Default is False)

    def set_inputs(self, inputs):
        self.config["prompt"] = inputs.get("prompt", "")
        self.config["use_prompt_enhancer"] = inputs.get("use_prompt_enhancer", False)  # Reset use_prompt_enhancer from clinet side.
        self.config["negative_prompt"] = inputs.get("negative_prompt", "")
        self.config["image_path"] = inputs.get("image_path", "")
        self.config["save_video_path"] = inputs.get("save_video_path", "")

    def run_input_encoder(self):
        image_encoder_output = None
        if self.config["task"] == "i2v":
            with ProfilingContext("Run Img Encoder"):
                image_encoder_output = self.run_image_encoder(self.config, self.image_encoder, self.vae_model)
        with ProfilingContext("Run Text Encoder"):
            prompt = self.config["prompt_enhanced"] if self.config["use_prompt_enhancer"] else self.config["prompt"]
            text_encoder_output = self.run_text_encoder(prompt, self.text_encoders, self.config, image_encoder_output)
        self.set_target_shape()
        self.inputs = {"text_encoder_output": text_encoder_output, "image_encoder_output": image_encoder_output}

        gc.collect()
        torch.cuda.empty_cache()

    @peak_memory_decorator
    def run(self):
        for step_index in range(self.model.scheduler.infer_steps):
            logger.info(f"==> step_index: {step_index + 1} / {self.model.scheduler.infer_steps}")

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
    @peak_memory_decorator
    def run_vae(self, latents, generator):
        images = self.vae_model.decode(latents, generator=generator, config=self.config)
        return images

    @ProfilingContext("Save video")
    def save_video(self, images):
        if not self.config.parallel_attn_type or (self.config.parallel_attn_type and dist.get_rank() == 0):
            if self.config.model_cls in ["wan2.1", "wan2.1_causvid", "wan2.1_skyreels_v2_df"]:
                cache_video(tensor=images, save_file=self.config.save_video_path, fps=self.config.get("fps", 16), nrow=1, normalize=True, value_range=(-1, 1))
            else:
                save_videos_grid(images, self.config.save_video_path, fps=self.config.get("fps", 24))

    def run_pipeline(self):
        if self.config["use_prompt_enhancer"]:
            self.config["prompt_enhanced"] = self.prompt_enhancer(self.config["prompt"])
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
