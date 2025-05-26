import torch


class BaseScheduler:
    def __init__(self, config):
        self.config = config
        self.latents = None
        self.step_index = 0
        self.infer_steps = config.infer_steps
        self.caching_records = [True] * config.infer_steps
        self.flag_df = False

    def step_pre(self, step_index):
        self.step_index = step_index
        self.latents = self.latents.to(dtype=torch.bfloat16)
