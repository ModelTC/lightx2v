from .utils import cache_init, cal_type
from ..scheduler import HunyuanScheduler
import torch


class HunyuanSchedulerTeaCaching(HunyuanScheduler):
    def __init__(self, config):
        super().__init__(config)
        self.teacache_thresh = self.config.teacache_thresh
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.previous_residual = None
        self.coefficients = [7.33226126e02, -4.01131952e02, 6.75869174e01, -3.14987800e00, 9.61237896e-02]

    def clear(self):
        if self.previous_residual is not None:
            self.previous_residual = self.previous_residual.cpu()
        if self.previous_modulated_input is not None:
            self.previous_modulated_input = self.previous_modulated_input.cpu()

        self.previous_modulated_input = None
        self.previous_residual = None
        torch.cuda.empty_cache()


class HunyuanSchedulerTaylorCaching(HunyuanScheduler):
    def __init__(self, config):
        super().__init__(config)
        self.cache_dic, self.current = cache_init(self.infer_steps)

    def step_pre(self, step_index):
        super().step_pre(step_index)
        self.current["step"] = step_index
        cal_type(self.cache_dic, self.current)
