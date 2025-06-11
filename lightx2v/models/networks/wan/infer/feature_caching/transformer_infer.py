from ..transformer_infer import BaseWanTransformerInfer
import torch
import numpy as np


# 1. TeaCaching
class WanTransformerInferTeaCaching(BaseWanTransformerInfer):
    # 1. 初始化
    def __init__(self, config):
        super().__init__(config)

        self.teacache_thresh = config.teacache_thresh
        self.accumulated_rel_l1_distance_even = 0
        self.previous_e0_even = None
        self.previous_residual_even = None
        self.accumulated_rel_l1_distance_odd = 0
        self.previous_e0_odd = None
        self.previous_residual_odd = None

        self.use_ret_steps = config.use_ret_steps
        self.set_attributes_by_task_and_model()
        self.cnt = 0

    def infer(self, weights, embed, grid_sizes, x, embed0, seq_lens, freqs, context):
        if self.infer_conditional:
            index = self.scheduler.step_index
            caching_records = self.scheduler.caching_records
            if index <= self.scheduler.infer_steps - 1:
                should_calc = self.calculate_should_calc(weights, embed, grid_sizes, x, embed0, seq_lens, freqs, context)
                self.scheduler.caching_records[index] = should_calc

            if caching_records[index]:
                x = self.infer_calculating(weights, grid_sizes, x, embed0, seq_lens, freqs, context)
            else:
                x = self.infer_using_cache(weights, grid_sizes, x, embed0, seq_lens, freqs, context)

        else:
            index = self.scheduler.step_index
            caching_records_2 = self.scheduler.caching_records_2
            if index <= self.scheduler.infer_steps - 1:
                should_calc = self.calculate_should_calc(weights, embed, grid_sizes, x, embed0, seq_lens, freqs, context)
                self.scheduler.caching_records_2[index] = should_calc

            if caching_records_2[index]:
                x = self.infer_calculating(weights, grid_sizes, x, embed0, seq_lens, freqs, context)
            else:
                x = self.infer_using_cache(weights, grid_sizes, x, embed0, seq_lens, freqs, context)
        
        if self.config.enable_cfg:
            super().switch_status()

        self.cnt += 1

        return x

    def infer_calculating(self, weights, grid_sizes, x, embed0, seq_lens, freqs, context):
        ori_x = x.clone()

        for block_idx in range(self.blocks_num):
            y_out, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = super().infer_block_1(weights.blocks[block_idx], grid_sizes, x, embed0, seq_lens, freqs, context)
            attn_out = super().infer_block_2(weights.blocks[block_idx], grid_sizes, x, embed0, seq_lens, freqs, context, y_out, gate_msa)
            y_out = super().infer_block_3(weights.blocks[block_idx], grid_sizes, x, embed0, seq_lens, freqs, context, attn_out, c_shift_msa, c_scale_msa)
            x = super().infer_block_4(weights.blocks[block_idx], grid_sizes, x, embed0, seq_lens, freqs, context, y_out, c_gate_msa)

        if self.infer_conditional:
            self.previous_residual_even = x - ori_x
        else:
            self.previous_residual_odd = x - ori_x
        return x

    def infer_using_cache(self, weights, grid_sizes, x, embed0, seq_lens, freqs, context):
        if self.infer_conditional:
            x += self.previous_residual_even
        else:
            x += self.previous_residual_odd
        return x

    # only in Wan2.1 TeaCaching
    def set_attributes_by_task_and_model(self):
        if self.config.task == "i2v":
            if self.use_ret_steps:
                if self.config.target_width == 480 or self.config.target_height == 480:
                    self.coefficients = [
                        2.57151496e05,
                        -3.54229917e04,
                        1.40286849e03,
                        -1.35890334e01,
                        1.32517977e-01,
                    ]
                if self.config.target_width == 720 or self.config.target_height == 720:
                    self.coefficients = [
                        8.10705460e03,
                        2.13393892e03,
                        -3.72934672e02,
                        1.66203073e01,
                        -4.17769401e-02,
                    ]
                self.ret_steps = 5 * 2
                self.cutoff_steps = self.config.infer_steps * 2
            else:
                if self.config.target_width == 480 or self.config.target_height == 480:
                    self.coefficients = [
                        -3.02331670e02,
                        2.23948934e02,
                        -5.25463970e01,
                        5.87348440e00,
                        -2.01973289e-01,
                    ]
                if self.config.target_width == 720 or self.config.target_height == 720:
                    self.coefficients = [
                        -114.36346466,
                        65.26524496,
                        -18.82220707,
                        4.91518089,
                        -0.23412683,
                    ]
                self.ret_steps = 1 * 2
                self.cutoff_steps = self.config.infer_steps * 2 - 2

        elif self.config.task == "t2v":
            if self.use_ret_steps:
                if "1.3B" in self.config.model_path:
                    self.coefficients = [-5.21862437e04, 9.23041404e03, -5.28275948e02, 1.36987616e01, -4.99875664e-02]
                if "14B" in self.config.model_path:
                    self.coefficients = [-3.03318725e05, 4.90537029e04, -2.65530556e03, 5.87365115e01, -3.15583525e-01]
                self.ret_steps = 5 * 2
                self.cutoff_steps = self.config.infer_steps * 2
            else:
                if "1.3B" in self.config.model_path:
                    self.coefficients = [2.39676752e03, -1.31110545e03, 2.01331979e02, -8.29855975e00, 1.37887774e-01]
                if "14B" in self.config.model_path:
                    self.coefficients = [-5784.54975374, 5449.50911966, -1811.16591783, 256.27178429, -13.02252404]
                self.ret_steps = 1 * 2
                self.cutoff_steps = self.config.infer_steps * 2 - 2

    # calculate should_calc
    def calculate_should_calc(self, weights, embed, grid_sizes, x, embed0, seq_lens, freqs, context):
        # 1. 时间步嵌入调制
        modulated_inp = embed0 if self.use_ret_steps else embed

        # 2. L1距离计算
        should_calc = False
        if self.infer_conditional:
            if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                should_calc = True
                self.accumulated_rel_l1_distance_even = 0
            else:
                rescale_func = np.poly1d(self.coefficients)
                self.accumulated_rel_l1_distance_even += rescale_func(
                    ((modulated_inp - self.previous_e0_even).abs().mean() / self.previous_e0_even.abs().mean()).cpu().item()
                )
                if self.accumulated_rel_l1_distance_even < self.teacache_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance_even = 0
            self.previous_e0_even = modulated_inp.clone()

        else:
            if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                should_calc = True
                self.accumulated_rel_l1_distance_odd = 0
            else:
                rescale_func = np.poly1d(self.coefficients)
                self.accumulated_rel_l1_distance_odd += rescale_func(
                    ((modulated_inp - self.previous_e0_odd).abs().mean() / self.previous_e0_odd.abs().mean()).cpu().item()
                )
                if self.accumulated_rel_l1_distance_odd < self.teacache_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance_odd = 0
            self.previous_e0_odd = modulated_inp.clone()

        # 3. 返回判断
        return should_calc
    

class WanTransformerInferTaylorCaching(BaseWanTransformerInfer):
    def __init__(self, config):
        super().__init__(config)

        self.blocks_cache_even = [{} for _ in range(self.blocks_num)]
        self.blocks_cache_odd = [{} for _ in range(self.blocks_num)]
        
    def infer(self, weights, embed, grid_sizes, x, embed0, seq_lens, freqs, context):
        if self.infer_conditional:
            index = self.scheduler.step_index
            caching_records = self.scheduler.caching_records

            if caching_records[index]:
                x = self.infer_calculating(weights, grid_sizes, x, embed0, seq_lens, freqs, context)
            else:
                x = self.infer_using_cache(weights, grid_sizes, x, embed0, seq_lens, freqs, context)

        else:
            index = self.scheduler.step_index
            caching_records_2 = self.scheduler.caching_records_2

            if caching_records_2[index]:
                x = self.infer_calculating(weights, grid_sizes, x, embed0, seq_lens, freqs, context)
            else:
                x = self.infer_using_cache(weights, grid_sizes, x, embed0, seq_lens, freqs, context)

        if self.config.enable_cfg:
            super().switch_status()

        return x

    def infer_calculating(self, weights, grid_sizes, x, embed0, seq_lens, freqs, context):
        for block_idx in range(self.blocks_num):
            y_out, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = super().infer_block_1(weights.blocks[block_idx], grid_sizes, x, embed0, seq_lens, freqs, context)
            if self.infer_conditional:
                super().derivative_approximation(self.blocks_cache_even[block_idx],"self_attn_out",y_out)
            else:
                super().derivative_approximation(self.blocks_cache_odd[block_idx],"self_attn_out",y_out)

            attn_out = super().infer_block_2(weights.blocks[block_idx], grid_sizes, x, embed0, seq_lens, freqs, context, y_out, gate_msa)
            if self.infer_conditional:
                super().derivative_approximation(self.blocks_cache_even[block_idx],"cross_attn_out",attn_out)
            else:
                super().derivative_approximation(self.blocks_cache_odd[block_idx],"cross_attn_out",attn_out)

            y_out = super().infer_block_3(weights.blocks[block_idx], grid_sizes, x, embed0, seq_lens, freqs, context, attn_out, c_shift_msa, c_scale_msa)
            if self.infer_conditional:
                super().derivative_approximation(self.blocks_cache_even[block_idx],"ffn_out",y_out)
            else:
                super().derivative_approximation(self.blocks_cache_odd[block_idx],"ffn_out",y_out)

            x = super().infer_block_4(weights.blocks[block_idx], grid_sizes, x, embed0, seq_lens, freqs, context, y_out, c_gate_msa)
        return x

    def infer_using_cache(self, weights, grid_sizes, x, embed0, seq_lens, freqs, context):
        for block_idx in range(self.blocks_num):
            x = self.infer_block(weights.blocks[block_idx], grid_sizes, x, embed0, seq_lens, freqs, context, block_idx)
        return x

    # 1. taylor using caching
    def infer_block(self, weights, grid_sizes, x, embed0, seq_lens, freqs, context, i):
        # 1. shift, scale, gate
        if embed0.dim() == 3:
            modulation = weights.modulation.tensor.unsqueeze(2)
            embed0 = (modulation + embed0).chunk(6, dim=1)
            _, _, gate_msa, _, _, c_gate_msa = [ei.squeeze(1) for ei in embed0]
        elif embed0.dim() == 2:
            _, _, gate_msa, _, _, c_gate_msa = (weights.modulation.tensor + embed0).chunk(6, dim=1)

        # 2. residual and taylor
        if self.infer_conditional:
            out = super().taylor_formula(self.blocks_cache_even[i]["self_attn_out"])
            out = out * gate_msa.squeeze(0)
            x = x + out

            out = super().taylor_formula(self.blocks_cache_even[i]["cross_attn_out"])
            x = x + out

            out = super().taylor_formula(self.blocks_cache_even[i]["ffn_out"])
            out = out * c_gate_msa.squeeze(0)
            x = x + out

        else:
            out = super().taylor_formula(self.blocks_cache_odd[i]["self_attn_out"])
            out = out * gate_msa.squeeze(0)
            x = x + out

            out = super().taylor_formula(self.blocks_cache_odd[i]["cross_attn_out"])
            x = x + out

            out = super().taylor_formula(self.blocks_cache_odd[i]["ffn_out"])
            out = out * c_gate_msa.squeeze(0)
            x = x + out

        return x
