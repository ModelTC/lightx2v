import torch


class BaseScheduler:
    def __init__(self, args):
        self.args = args
        self.step_index = 0
        self.latents = None

    def step_pre(self, step_index):
        self.step_index = step_index
        self.latents = self.latents.to(dtype=torch.bfloat16)

    def clear(self):
        pass
