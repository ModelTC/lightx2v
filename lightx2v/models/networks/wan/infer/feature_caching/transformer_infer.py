from ..transformer_infer import BaseWanTransformerInfer
import torch
import numpy as np


# 1. TeaCaching
class WanTransformerInferTeaCaching(BaseWanTransformerInfer):
    # 1. 初始化
    def __init__(self, config):
        super().__init__(config)

        self.status = "even"
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
        if self.status == "even":
            index = self.scheduler.step_index
            caching_records = self.scheduler.caching_records
            if index <= self.scheduler.infer_steps - 1:
                should_calc = self.calculate_should_calc(weights, embed, grid_sizes, x, embed0, seq_lens, freqs, context)
                self.scheduler.caching_records[index] = should_calc

            if caching_records[index]:
                x = self.infer_calculating(weights, grid_sizes, x, embed0, seq_lens, freqs, context)
            else:
                x = self.infer_using_cache(weights, grid_sizes, x, embed0, seq_lens, freqs, context)
            self.status = "odd"

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
            self.status = "even"

        self.cnt += 1

        return x


    def infer_calculating(self, weights, grid_sizes, x, embed0, seq_lens, freqs, context):
        ori_x = x.clone()

        for block_idx in range(self.blocks_num):
            y_out, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = super().infer_block_1(weights.blocks[block_idx], grid_sizes, x, embed0, seq_lens, freqs, context)
            attn_out = super().infer_block_2(weights.blocks[block_idx], grid_sizes, x, embed0, seq_lens, freqs, context, y_out, gate_msa)
            y_out = super().infer_block_3(weights.blocks[block_idx], grid_sizes, x, embed0, seq_lens, freqs, context, attn_out, c_shift_msa, c_scale_msa)
            x = super().infer_block_4(weights.blocks[block_idx], grid_sizes, x, embed0, seq_lens, freqs, context, y_out, c_gate_msa)

        if self.status == "even":
            self.previous_residual_even = x - ori_x
        else:
            self.previous_residual_odd = x - ori_x
        return x

    def infer_using_cache(self, weights, grid_sizes, x, embed0, seq_lens, freqs, context):
        if self.status == "even":
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
        if self.status == "even":
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

        self.status = "even"
        self.blocks_cache_even = [{} for _ in range(self.blocks_num)]
        self.blocks_cache_odd = [{} for _ in range(self.blocks_num)]
        
    def infer(self, weights, embed, grid_sizes, x, embed0, seq_lens, freqs, context):
        if self.status == "even":
            index = self.scheduler.step_index
            caching_records = self.scheduler.caching_records

            if caching_records[index]:
                x = self.infer_calculating(weights, grid_sizes, x, embed0, seq_lens, freqs, context)
            else:
                x = self.infer_using_cache(weights, grid_sizes, x, embed0, seq_lens, freqs, context)
            self.status = "odd"

        else:
            index = self.scheduler.step_index
            caching_records_2 = self.scheduler.caching_records_2

            if caching_records_2[index]:
                x = self.infer_calculating(weights, grid_sizes, x, embed0, seq_lens, freqs, context)
            else:
                x = self.infer_using_cache(weights, grid_sizes, x, embed0, seq_lens, freqs, context)
            self.status = "even"

        return x

    def infer_calculating(self, weights, grid_sizes, x, embed0, seq_lens, freqs, context):
        for block_idx in range(self.blocks_num):
            y_out, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = super().infer_block_1(weights.blocks[block_idx], grid_sizes, x, embed0, seq_lens, freqs, context)
            if self.status == "even":
                self.derivative_approximation(self.blocks_cache_even[block_idx],"self_attn_out",y_out)
            else:
                self.derivative_approximation(self.blocks_cache_odd[block_idx],"self_attn_out",y_out)

            attn_out = super().infer_block_2(weights.blocks[block_idx], grid_sizes, x, embed0, seq_lens, freqs, context, y_out, gate_msa)
            if self.status == "even":
                self.derivative_approximation(self.blocks_cache_even[block_idx],"cross_attn_out",attn_out)
            else:
                self.derivative_approximation(self.blocks_cache_odd[block_idx],"cross_attn_out",attn_out)

            y_out = super().infer_block_3(weights.blocks[block_idx], grid_sizes, x, embed0, seq_lens, freqs, context, attn_out, c_shift_msa, c_scale_msa)
            if self.status == "even":
                self.derivative_approximation(self.blocks_cache_even[block_idx],"ffn_out",y_out)
            else:
                self.derivative_approximation(self.blocks_cache_odd[block_idx],"ffn_out",y_out)

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
        if self.status == "even":
            out = self.taylor_formula(self.blocks_cache_even[i]["self_attn_out"])
            out = out * gate_msa.squeeze(0)
            x = x + out

            out = self.taylor_formula(self.blocks_cache_even[i]["cross_attn_out"])
            x = x + out

            out = self.taylor_formula(self.blocks_cache_even[i]["ffn_out"])
            out = out * c_gate_msa.squeeze(0)
            x = x + out

            return x

        else:
            out = self.taylor_formula(self.blocks_cache_odd[i]["self_attn_out"])
            out = out * gate_msa.squeeze(0)
            x = x + out

            out = self.taylor_formula(self.blocks_cache_odd[i]["cross_attn_out"])
            x = x + out

            out = self.taylor_formula(self.blocks_cache_odd[i]["ffn_out"])
            out = out * c_gate_msa.squeeze(0)
            x = x + out

            return x

    # 1. when fully calcualted, stored in cache
    def derivative_approximation(self, block_cache, module_name, out):
        if module_name not in block_cache:
            block_cache[module_name] = {0: out}
        else:
            step_diff = self.get_taylor_step_diff()

            previous_out = block_cache[module_name][0]
            block_cache[module_name][0] = out
            block_cache[module_name][1] = (out - previous_out) / step_diff

    def get_taylor_step_diff(self):
        step_diff = 0
        if self.status == "even":
            current_step = self.scheduler.step_index
            last_calc_step = current_step - 1
            while last_calc_step >= 0 and not self.scheduler.caching_records[last_calc_step]:
                last_calc_step -= 1
            step_diff = current_step - last_calc_step
        else:
            current_step = self.scheduler.step_index
            last_calc_step = current_step - 1
            while last_calc_step >= 0 and not self.scheduler.caching_records_2[last_calc_step]:
                last_calc_step -= 1
            step_diff = current_step - last_calc_step
        
        return step_diff
    
    def taylor_formula(self, tensor_dict):
        x = self.get_taylor_step_diff()

        output = 0
        for i in range(len(tensor_dict)):
            output += (1 / math.factorial(i)) * tensor_dict[i] * (x**i)

        return output
    

class TransformerArgs:
    def __init__(self, config):
        # 1. cache
        self.previous_residual_tiny = None
        self.now_residual_tiny = None
        self.norm_ord = 1
        self.skipped_step_length = 1
        self.previous_residual = None

        # 2. moreg
        self.previous_moreg = 1.
        self.moreg_strides = [1]
        self.moreg_steps = [
            int(0.1 * config.infer_steps),
            int(0.9 * config.infer_steps)
        ]
        self.moreg_hyp = [0.385, 8, 1, 2]
        self.mograd_mul = 10
        self.spatial_dim = 1536


class WanTransformerInferAdaCaching(BaseWanTransformerInfer):
    def __init__(self, config):
        # super().__init__(config)
        # # 1. fixed args
        # self.decisive_double_block_id = config.infer_steps/2
        # self.codebook = {0.03: 12, 0.05: 10, 0.07: 8, 0.09: 6, 0.11: 4, 1.00: 3}

        # # 2. cache + moreg
        # self.status = "even"
        # self.args_even = TransformerArgs(config)
        # self.args_odd = TransformerArgs(config) 
        pass

    def infer(self, weights, embed, grid_sizes, x, embed0, seq_lens, freqs, context):
        # if self.status == "even":
        #     index = self.scheduler.step_index
        #     caching_records = self.scheduler.caching_records

        #     if caching_records[index]:
        #         x = self.infer_calculating(weights, grid_sizes, x, embed0, seq_lens, freqs, context)
        #     else:
        #         x = self.infer_using_cache(weights, grid_sizes, x, embed0, seq_lens, freqs, context)
        #     self.status = "odd"

        # else:
        #     index = self.scheduler.step_index
        #     caching_records_2 = self.scheduler.caching_records_2

        #     if caching_records_2[index]:
        #         x = self.infer_calculating(weights, grid_sizes, x, embed0, seq_lens, freqs, context)
        #     else:
        #         x = self.infer_using_cache(weights, grid_sizes, x, embed0, seq_lens, freqs, context)
        #     self.status = "even"

        # return x
        pass
    
    def infer_calculating(self, weights, grid_sizes, x, embed0, seq_lens, freqs, context):
        # if self.status == "even":
        # else:
        # return x
        pass
