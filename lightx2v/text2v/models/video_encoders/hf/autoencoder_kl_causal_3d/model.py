import os
import torch
from .autoencoder_kl_causal_3d import AutoencoderKLCausal3D


class VideoEncoderKLCausal3DModel:
    def __init__(self, model_path, dtype, device):
        self.model_path = model_path
        self.dtype = dtype
        self.device = device
        self.load()

    def load(self):
        self.vae_path = os.path.join(self.model_path, "hunyuan-video-t2v-720p/vae")
        config = AutoencoderKLCausal3D.load_config(self.vae_path)
        self.model = AutoencoderKLCausal3D.from_config(config)
        ckpt = torch.load(os.path.join(self.vae_path, "pytorch_model.pt"), map_location="cpu", weights_only=True)
        self.model.load_state_dict(ckpt)
        self.model = self.model.to(dtype=self.dtype, device=self.device)
        self.model.requires_grad_(False)
        self.model.eval()

    def to_cpu(self):
        self.model = self.model.to("cpu")

    def to_cuda(self):
        self.model = self.model.to("cuda")

    def decode(self, latents, generator, args):
        if args.cpu_offload:
            self.to_cuda()
        latents = latents / self.model.config.scaling_factor
        latents = latents.to(dtype=self.dtype, device=torch.device("cuda"))
        self.model.enable_tiling()
        image = self.model.decode(latents, return_dict=False, generator=generator)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().float()
        if args.cpu_offload:
            self.to_cpu()
        return image


if __name__ == "__main__":
    vae_model = VideoEncoderKLCausal3DModel(
        "/mnt/nvme0/yongyang/projects/hy/new/HunyuanVideo/ckpts",
        dtype=torch.float16,
        device=torch.device("cuda"),
    )
