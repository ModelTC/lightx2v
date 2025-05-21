import os
import math
import numpy as np
import torch

from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from loguru import logger


class WanCausVidDfScheduler(WanScheduler):
    def __init__(self, config):

        total_latens_frames_num = config.num_frame_per_block + (
                                  (config.num_frame_per_block - config.block_overlap_frame) 
                                   * (config.num_blocks - 1))
        config.total_latens_frames_num = total_latens_frames_num
        config.target_video_length = total_latens_frames_num * 4 + 1

        super().__init__(config)
        self.overlap = config.block_overlap_frame
        self.flag_df = True
        self.df_timesteps = None

    def prepare(self, image_encoder_output=None):
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.config.seed)

        self.prepare_latents(self.config.target_shape, dtype=torch.float32)

        if os.path.isfile(self.config.image_path):
            self.seq_len = self.config.lat_h * self.config.lat_w // (self.config.patch_size[1] * self.config.patch_size[2]) * self.config.target_shape[1]
        else:
            self.seq_len = math.ceil((self.config.target_shape[2] * self.config.target_shape[3]) / (self.config.patch_size[1] * self.config.patch_size[2]) * self.config.target_shape[1])

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

        self.set_timesteps(self.infer_steps, device=self.device, shift=self.sample_shift)

        self.num_frame_per_block = self.config.num_frame_per_block

        self.df_schedulers = []
        old_shape = self.config.target_shape
        for _ in range(self.num_frame_per_block):
            sample_scheduler = WanScheduler(self.config)
            new_shape = (old_shape[0], 1, *old_shape[2:])  # 把第1维改成1，保持其他维度不变
            sample_scheduler.config.target_shape = new_shape

            sample_scheduler.prepare()
            self.df_schedulers.append(sample_scheduler)
        #复原总的scheduler target_shape
        self.config.target_shape = old_shape

    def reset(self):
        self.model_outputs = [None] * self.solver_order
        self.timestep_list = [None] * self.solver_order
        self.last_sample = None
        self.noise_pred = None
        self.this_order = None
        self.lower_order_nums = 0
        self.prepare_latents(self.config.target_shape, dtype=torch.float32)

        self.df_timesteps = self.timesteps.unsqueeze(1).repeat(1, self.num_frame_per_block)
        
        for scheduler in self.df_schedulers:
            scheduler.reset()

        
    def step_pre(self, step_index, block_idx):
        self.step_index = step_index
        self.block_idx  = block_idx
        if self.step_index == 0:#每一个block开始时候重置
            if block_idx > 0:
                self.pre_block_tail = self.latents[:, -self.overlap:]

            self.reset()

            if self.block_idx > 0 and self.overlap > 0:
                self.df_timesteps[:, :self.overlap] = 0

        self.latents    = self.latents.to(dtype=torch.bfloat16)
        
        if block_idx > 0 and self.overlap > 0:
            self.latents[:, :self.overlap] = self.pre_block_tail

    def step_post(self):
        if self.block_idx == 0:
            st = 0
        else:
            st = self.overlap

        for idx in range(st, 7):  # 每一帧单独step
            self.df_schedulers[idx].step_pre(step_index=self.step_index)
            self.df_schedulers[idx].noise_pred = self.noise_pred[:, idx]
            self.df_schedulers[idx].timesteps[self.step_index] = self.df_timesteps[self.step_index, idx]
            self.df_schedulers[idx].latents = self.latents[:, idx]
            self.df_schedulers[idx].step_post()

            self.latents[:, idx] = self.df_schedulers[idx].latents
