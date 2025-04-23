import gc
import torch
import torch.distributed as dist
from lightx2v.utils.profiler import ProfilingContext4Debug, ProfilingContext
from lightx2v.utils.utils import save_videos_grid, cache_video
from lightx2v.utils.envs import *


class DefaultRunner:
    def __init__(self, config):
        self.config = config
        self.model, self.text_encoders, self.vae_model, self.image_encoder = self.load_model()

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
        if self.config.cpu_offload:
            self.model.scheduler.clear()
            del self.inputs, self.model.scheduler, self.model, self.text_encoders
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
        self.init_scheduler()
        self.run_input_encoder()
        self.model.scheduler.prepare(self.inputs["image_encoder_output"])
        latents, generator = self.run()
        self.end_run()
        images = self.run_vae(latents, generator)
        self.save_video(images)
