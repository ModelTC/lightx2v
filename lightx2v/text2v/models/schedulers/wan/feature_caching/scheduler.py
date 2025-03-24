import torch
from ..scheduler import WanScheduler


class WanSchedulerFeatureCaching(WanScheduler):
    def __init__(self, args):
        super().__init__(args)
        self.cnt = 0
        self.num_steps = self.args.infer_steps * 2
        self.teacache_thresh = self.args.teacache_thresh
        self.accumulated_rel_l1_distance_even = 0
        self.accumulated_rel_l1_distance_odd = 0
        self.previous_e0_even = None
        self.previous_e0_odd = None
        self.previous_residual_even = None
        self.previous_residual_odd = None
        self.use_ret_steps = self.args.use_ret_steps

        if self.use_ret_steps:
            if self.args.target_width == 480 or self.args.target_height == 480:
                self.coefficients = [
                    2.57151496e05,
                    -3.54229917e04,
                    1.40286849e03,
                    -1.35890334e01,
                    1.32517977e-01,
                ]
            if self.args.target_width == 720 or self.args.target_height == 720:
                self.coefficients = [
                    8.10705460e03,
                    2.13393892e03,
                    -3.72934672e02,
                    1.66203073e01,
                    -4.17769401e-02,
                ]
            self.ret_steps = 5 * 2
            self.cutoff_steps = self.args.infer_steps * 2
        else:
            if self.args.target_width == 480 or self.args.target_height == 480:
                self.coefficients = [
                    -3.02331670e02,
                    2.23948934e02,
                    -5.25463970e01,
                    5.87348440e00,
                    -2.01973289e-01,
                ]
            if self.args.target_width == 720 or self.args.target_height == 720:
                self.coefficients = [
                    -114.36346466,
                    65.26524496,
                    -18.82220707,
                    4.91518089,
                    -0.23412683,
                ]
            self.ret_steps = 1 * 2
            self.cutoff_steps = self.args.infer_steps * 2 - 2