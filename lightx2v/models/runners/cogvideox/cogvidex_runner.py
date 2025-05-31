from diffusers.utils import export_to_video, load_image
import imageio
import numpy as np
import torch

from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.utils.profiler import ProfilingContext
from lightx2v.models.input_encoders.hf.t5_v1_1_xxl.model import T5EncoderModel_v1_1_xxl
from lightx2v.models.networks.cogvideox.model import CogvideoxModel
from lightx2v.models.video_encoders.hf.cogvideox.model import CogvideoxVAE
from lightx2v.models.schedulers.cogvideox.scheduler import CogvideoxXDPMScheduler


@RUNNER_REGISTER("cogvideox")
class CogvideoxRunner(DefaultRunner):
    def __init__(self, config):
        super().__init__(config)

    @ProfilingContext("Load models")
    def load_model(self):
        text_encoder = T5EncoderModel_v1_1_xxl(self.config)
        text_encoders = [text_encoder]
        model = CogvideoxModel(self.config)
        vae_model = CogvideoxVAE(self.config)
        image_encoder = None
        return model, text_encoders, vae_model, image_encoder

    def init_scheduler(self):
        scheduler = CogvideoxXDPMScheduler(self.config)
        self.model.set_scheduler(scheduler)

    def run_text_encoder(self, text, text_encoders, config, image_encoder_output):
        text_encoder_output = {}
        n_prompt = config.get("negative_prompt", "")
        context = text_encoders[0].infer([text], config)
        context_null = text_encoders[0].infer([n_prompt if n_prompt else ""], config)
        text_encoder_output["context"] = context
        text_encoder_output["context_null"] = context_null
        return text_encoder_output

    def run_image_encoder(self, config, image_encoder, vae_model):
        image = load_image(image=self.config.image_path)
        image = vae_model.video_processor.preprocess(image, height=config.height, width=config.width).to(torch.device("cuda"), dtype=torch.bfloat16)
        image = image.unsqueeze(2)
        image = [vae_model.encode(img.unsqueeze(0)) for img in image]
        return image

    def set_target_shape(self):
        num_frames = self.config.target_video_length
        latent_frames = (num_frames - 1) // self.config.vae_scale_factor_temporal + 1
        additional_frames = 0
        patch_size_t = self.config.patch_size_t
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            additional_frames = patch_size_t - latent_frames % patch_size_t
            num_frames += additional_frames * self.config.vae_scale_factor_temporal
        target_shape = (
            self.config.batch_size,
            (num_frames - 1) // self.config.vae_scale_factor_temporal + 1,
            self.config.latent_channels,
            self.config.height // self.config.vae_scale_factor_spatial,
            self.config.width // self.config.vae_scale_factor_spatial,
        )
        if self.config.task in ["t2v"]:
            self.config.target_shape = target_shape
        elif self.config.task in ["i2v"]:
            self.config.target_shape = target_shape[:1] + (target_shape[1] + target_shape[1] % self.config.patch_size_t,) + target_shape[2:]
            self.config.padding_shape = (
                self.config.batch_size,
                (num_frames - 1) // self.config.vae_scale_factor_temporal,
                self.config.latent_channels,
                self.config.height // self.config.vae_scale_factor_spatial,
                self.config.width // self.config.vae_scale_factor_spatial,
            )

    def save_video(self, images):
        with imageio.get_writer(self.config.save_video_path, fps=16) as writer:
            for pil_image in images:
                frame_np = np.array(pil_image, dtype=np.uint8)
                writer.append_data(frame_np)
