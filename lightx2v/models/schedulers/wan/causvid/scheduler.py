import math
import numpy as np
import torch
from typing import List, Optional, Tuple, Union
from lightx2v.models.schedulers.wan.scheduler import WanScheduler


class WanCausVidScheduler(WanScheduler):
    def __init__(self, config):
        super().__init__(config)
        self.denoising_step_list = config.denoising_step_list
        self.infer_steps = self.config.infer_steps
        self.sample_shift = self.config.sample_shift

    def prepare(self, image_encoder_output=None):
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.config.seed)

        self.prepare_latents(self.config.target_shape, dtype=torch.float32)

        if self.config.task in ["t2v"]:
            self.seq_len = math.ceil((self.config.target_shape[2] * self.config.target_shape[3]) / (self.config.patch_size[1] * self.config.patch_size[2]) * self.config.target_shape[1])
        elif self.config.task in ["i2v"]:
            self.seq_len = self.config.lat_h * self.config.lat_w // (self.config.patch_size[1] * self.config.patch_size[2]) * self.config.target_shape[1]

        alphas = np.linspace(1, 1 / self.num_train_timesteps, self.num_train_timesteps)[::-1].copy()
        sigmas = 1.0 - alphas
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32)

        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

        self.sigmas = sigmas
        self.timesteps = sigmas * self.num_train_timesteps

        self.model_outputs = [None] * self.solver_order
        self.timestep_list = [None] * self.solver_order
        self.last_sample = None

        self.sigmas = self.sigmas.to("cpu")
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

        if len(self.denoising_step_list) == self.infer_steps:  # 如果denoising_step_list有效既使用
            self.set_denoising_timesteps(device=self.device)
        else:
            self.set_timesteps(self.infer_steps, device=self.device, shift=self.sample_shift)

    def set_denoising_timesteps(self, device: Union[str, torch.device] = None):
        self.timesteps = torch.tensor(self.denoising_step_list, device=device, dtype=torch.int64)
        self.sigmas = torch.cat([self.timesteps / self.num_train_timesteps, torch.tensor([0.0], device=device)])
        self.sigmas = self.sigmas.to("cpu")
        self.infer_steps = len(self.timesteps)

        self.model_outputs = [
            None,
        ] * self.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self._begin_index = None

    def reset(self):
        self.model_outputs = [None] * self.solver_order
        self.timestep_list = [None] * self.solver_order
        self.last_sample = None
        self.noise_pred = None
        self.this_order = None
        self.lower_order_nums = 0
        self.prepare_latents(self.config.target_shape, dtype=torch.float32)

    def step_pre(self, step_index, block_idx=None):
        self.step_index = step_index
        if self.step_index == 0:#每一个block开始时候重置
            self.reset()
        self.latents    = self.latents.to(dtype=torch.bfloat16)