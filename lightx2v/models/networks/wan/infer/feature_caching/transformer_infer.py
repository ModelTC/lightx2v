from ..transformer_infer import WanTransformerInfer
from lightx2v.common.transformer_infer.transformer_infer import BaseTaylorCachingTransformerInfer
import torch
import numpy as np


# 1. TeaCaching
class WanTransformerInferTeaCaching(WanTransformerInfer):
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
    def calculate_should_calc(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        # 1. timestep embedding
        modulated_inp = embed0 if self.use_ret_steps else embed

        # 2. L1 calculate
        should_calc = False
        if self.infer_conditional:
            if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                should_calc = True
                self.accumulated_rel_l1_distance_even = 0
            else:
                rescale_func = np.poly1d(self.coefficients)
                self.accumulated_rel_l1_distance_even += rescale_func(((modulated_inp - self.previous_e0_even).abs().mean() / self.previous_e0_even.abs().mean()).cpu().item())
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
                self.accumulated_rel_l1_distance_odd += rescale_func(((modulated_inp - self.previous_e0_odd).abs().mean() / self.previous_e0_odd.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance_odd < self.teacache_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance_odd = 0
            self.previous_e0_odd = modulated_inp.clone()

        # 3. return the judgement
        return should_calc

    def infer(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        if self.infer_conditional:
            index = self.scheduler.step_index
            caching_records = self.scheduler.caching_records
            if index <= self.scheduler.infer_steps - 1:
                should_calc = self.calculate_should_calc(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)
                self.scheduler.caching_records[index] = should_calc

            if caching_records[index]:
                x = self.infer_calculating(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)
            else:
                x = self.infer_using_cache(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)

        else:
            index = self.scheduler.step_index
            caching_records_2 = self.scheduler.caching_records_2
            if index <= self.scheduler.infer_steps - 1:
                should_calc = self.calculate_should_calc(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)
                self.scheduler.caching_records_2[index] = should_calc

            if caching_records_2[index]:
                x = self.infer_calculating(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)
            else:
                x = self.infer_using_cache(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)

        if self.config.enable_cfg:
            self.switch_status()

        self.cnt += 1

        return x

    def infer_calculating(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        ori_x = x.clone()

        for block_idx in range(self.blocks_num):
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = self.infer_phase_1(weights.blocks[block_idx], grid_sizes, embed, x, embed0, seq_lens, freqs, context)
            y_out = self.infer_phase_2(weights.blocks[block_idx].compute_phases[0], grid_sizes, x, seq_lens, freqs, shift_msa, scale_msa)
            attn_out = self.infer_phase_3(weights.blocks[block_idx].compute_phases[1], x, context, y_out, gate_msa)
            y_out = self.infer_phase_4(weights.blocks[block_idx].compute_phases[2], x, attn_out, c_shift_msa, c_scale_msa)
            x = self.infer_phase_5(x, y_out, c_gate_msa)

        if self.infer_conditional:
            self.previous_residual_even = x - ori_x
        else:
            self.previous_residual_odd = x - ori_x
        return x

    def infer_using_cache(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        if self.infer_conditional:
            x += self.previous_residual_even
        else:
            x += self.previous_residual_odd
        return x

    def clear(self):
        if self.previous_residual_even is not None:
            self.previous_residual_even = self.previous_residual_even.cpu()
        if self.previous_residual_odd is not None:
            self.previous_residual_odd = self.previous_residual_odd.cpu()
        if self.previous_e0_even is not None:
            self.previous_e0_even = self.previous_e0_even.cpu()
        if self.previous_e0_odd is not None:
            self.previous_e0_odd = self.previous_e0_odd.cpu()

        self.previous_residual_even = None
        self.previous_residual_odd = None
        self.previous_e0_even = None
        self.previous_e0_odd = None

        torch.cuda.empty_cache()


class WanTransformerInferTaylorCaching(WanTransformerInfer, BaseTaylorCachingTransformerInfer):
    def __init__(self, config):
        super().__init__(config)

        self.blocks_cache_even = [{} for _ in range(self.blocks_num)]
        self.blocks_cache_odd = [{} for _ in range(self.blocks_num)]

    # 1. get taylor step_diff when there is two caching_records in scheduler
    def get_taylor_step_diff(self):
        step_diff = 0
        if self.infer_conditional:
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

    def infer(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        if self.infer_conditional:
            index = self.scheduler.step_index
            caching_records = self.scheduler.caching_records

            if caching_records[index]:
                x = self.infer_calculating(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)
            else:
                x = self.infer_using_cache(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)

        else:
            index = self.scheduler.step_index
            caching_records_2 = self.scheduler.caching_records_2

            if caching_records_2[index]:
                x = self.infer_calculating(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)
            else:
                x = self.infer_using_cache(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)

        if self.config.enable_cfg:
            self.switch_status()

        return x

    def infer_calculating(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        for block_idx in range(self.blocks_num):
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = self.infer_phase_1(weights.blocks[block_idx], grid_sizes, embed, x, embed0, seq_lens, freqs, context)

            y_out = self.infer_phase_2(weights.blocks[block_idx].compute_phases[0], grid_sizes, x, seq_lens, freqs, shift_msa, scale_msa)
            if self.infer_conditional:
                self.derivative_approximation(self.blocks_cache_even[block_idx], "self_attn_out", y_out)
            else:
                self.derivative_approximation(self.blocks_cache_odd[block_idx], "self_attn_out", y_out)

            attn_out = self.infer_phase_3(weights.blocks[block_idx].compute_phases[1], x, context, y_out, gate_msa)
            if self.infer_conditional:
                self.derivative_approximation(self.blocks_cache_even[block_idx], "cross_attn_out", attn_out)
            else:
                self.derivative_approximation(self.blocks_cache_odd[block_idx], "cross_attn_out", attn_out)

            y_out = self.infer_phase_4(weights.blocks[block_idx].compute_phases[2], x, attn_out, c_shift_msa, c_scale_msa)
            if self.infer_conditional:
                self.derivative_approximation(self.blocks_cache_even[block_idx], "ffn_out", y_out)
            else:
                self.derivative_approximation(self.blocks_cache_odd[block_idx], "ffn_out", y_out)

            x = self.infer_phase_5(x, y_out, c_gate_msa)
        return x

    def infer_using_cache(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        for block_idx in range(self.blocks_num):
            x = self.infer_block(weights.blocks[block_idx], grid_sizes, embed, x, embed0, seq_lens, freqs, context, block_idx)
        return x

    # 1. taylor using caching
    def infer_block(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context, i):
        # 1. shift, scale, gate
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = self.infer_phase_1(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)

        # 2. residual and taylor
        if self.infer_conditional:
            out = self.taylor_formula(self.blocks_cache_even[i]["self_attn_out"])
            out = out * gate_msa.squeeze(0)
            x = x + out

            out = self.taylor_formula(self.blocks_cache_even[i]["cross_attn_out"])
            x = x + out

            out = self.taylor_formula(self.blocks_cache_even[i]["ffn_out"])
            out = out * c_gate_msa.squeeze(0)
            x = x + out

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

    def clear(self):
        for cache in self.blocks_cache_even:
            for key in cache:
                if cache[key] is not None:
                    cache[key] = cache[key].cpu()
            cache.clear()

        for cache in self.blocks_cache_odd:
            for key in cache:
                if cache[key] is not None:
                    cache[key] = cache[key].cpu()
            cache.clear()
        torch.cuda.empty_cache()


class WanTransformerInferAdaCaching(WanTransformerInfer):
    def __init__(self, config):
        super().__init__(config)

        # 1. fixed args
        self.decisive_double_block_id = self.blocks_num // 2
        self.codebook = {0.03: 12, 0.05: 10, 0.07: 8, 0.09: 6, 0.11: 4, 1.00: 3}

        # 2. Create two instances of AdaArgs
        self.args_even = AdaArgs(config)
        self.args_odd = AdaArgs(config)

    def infer(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        if self.infer_conditional:
            index = self.scheduler.step_index
            caching_records = self.scheduler.caching_records

            if caching_records[index]:
                x = self.infer_calculating(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)

                # 1. calculate the skipped step length
                if index <= self.scheduler.infer_steps - 2:
                    self.args_even.skipped_step_length = self.calculate_skip_step_length()
                    for i in range(1, self.args_even.skipped_step_length):
                        if (index + i) <= self.scheduler.infer_steps - 1:
                            self.scheduler.caching_records[index + i] = False
            else:
                x = self.infer_using_cache(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)

        else:
            index = self.scheduler.step_index
            caching_records = self.scheduler.caching_records_2

            if caching_records[index]:
                x = self.infer_calculating(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)

                # 1. calculate the skipped step length
                if index <= self.scheduler.infer_steps - 2:
                    self.args_odd.skipped_step_length = self.calculate_skip_step_length()
                    for i in range(1, self.args_odd.skipped_step_length):
                        if (index + i) <= self.scheduler.infer_steps - 1:
                            self.scheduler.caching_records_2[index + i] = False
            else:
                x = self.infer_using_cache(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)

        if self.config.enable_cfg:
            self.switch_status()

        return x

    def infer_calculating(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        ori_x = x.clone()

        for block_idx in range(self.blocks_num):
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = self.infer_phase_1(weights.blocks[block_idx], grid_sizes, embed, x, embed0, seq_lens, freqs, context)

            y_out = self.infer_phase_2(weights.blocks[block_idx].compute_phases[0], grid_sizes, x, seq_lens, freqs, shift_msa, scale_msa)
            if block_idx == self.decisive_double_block_id:
                if self.infer_conditional:
                    self.args_even.now_residual_tiny = y_out * gate_msa.squeeze(0)
                else:
                    self.args_odd.now_residual_tiny = y_out * gate_msa.squeeze(0)

            attn_out = self.infer_phase_3(weights.blocks[block_idx].compute_phases[1], x, context, y_out, gate_msa)
            y_out = self.infer_phase_4(weights.blocks[block_idx].compute_phases[2], x, attn_out, c_shift_msa, c_scale_msa)
            x = self.infer_phase_5(x, y_out, c_gate_msa)

        if self.infer_conditional:
            self.args_even.previous_residual = x - ori_x
        else:
            self.args_odd.previous_residual = x - ori_x
        return x

    def infer_using_cache(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        if self.infer_conditional:
            x += self.args_even.previous_residual
        else:
            x += self.args_odd.previous_residual
        return x

    def calculate_skip_step_length(self):
        if self.infer_conditional:
            if self.args_even.previous_residual_tiny is None:
                self.args_even.previous_residual_tiny = self.args_even.now_residual_tiny
                return 1
            else:
                cache = self.args_even.previous_residual_tiny
                res = self.args_even.now_residual_tiny
                norm_ord = self.args_even.norm_ord
                cache_diff = (cache - res).norm(dim=(0, 1), p=norm_ord) / cache.norm(dim=(0, 1), p=norm_ord)
                cache_diff = cache_diff / self.args_even.skipped_step_length

                if self.args_even.moreg_steps[0] <= self.scheduler.step_index <= self.args_even.moreg_steps[1]:
                    moreg = 0
                    for i in self.args_even.moreg_strides:
                        moreg_i = (res[i * self.args_even.spatial_dim :, :] - res[: -i * self.args_even.spatial_dim, :]).norm(p=norm_ord)
                        moreg_i /= res[i * self.args_even.spatial_dim :, :].norm(p=norm_ord) + res[: -i * self.args_even.spatial_dim, :].norm(p=norm_ord)
                        moreg += moreg_i
                    moreg = moreg / len(self.args_even.moreg_strides)
                    moreg = ((1 / self.args_even.moreg_hyp[0] * moreg) ** self.args_even.moreg_hyp[1]) / self.args_even.moreg_hyp[2]
                else:
                    moreg = 1.0

                mograd = self.args_even.mograd_mul * (moreg - self.args_even.previous_moreg) / self.args_even.skipped_step_length
                self.args_even.previous_moreg = moreg
                moreg = moreg + abs(mograd)
                cache_diff = cache_diff * moreg

                metric_thres, cache_rates = list(self.codebook.keys()), list(self.codebook.values())
                if cache_diff < metric_thres[0]:
                    new_rate = cache_rates[0]
                elif cache_diff < metric_thres[1]:
                    new_rate = cache_rates[1]
                elif cache_diff < metric_thres[2]:
                    new_rate = cache_rates[2]
                elif cache_diff < metric_thres[3]:
                    new_rate = cache_rates[3]
                elif cache_diff < metric_thres[4]:
                    new_rate = cache_rates[4]
                else:
                    new_rate = cache_rates[-1]

                self.args_even.previous_residual_tiny = self.args_even.now_residual_tiny
                return new_rate

        else:
            if self.args_odd.previous_residual_tiny is None:
                self.args_odd.previous_residual_tiny = self.args_odd.now_residual_tiny
                return 1
            else:
                cache = self.args_odd.previous_residual_tiny
                res = self.args_odd.now_residual_tiny
                norm_ord = self.args_odd.norm_ord
                cache_diff = (cache - res).norm(dim=(0, 1), p=norm_ord) / cache.norm(dim=(0, 1), p=norm_ord)
                cache_diff = cache_diff / self.args_odd.skipped_step_length

                if self.args_odd.moreg_steps[0] <= self.scheduler.step_index <= self.args_odd.moreg_steps[1]:
                    moreg = 0
                    for i in self.args_odd.moreg_strides:
                        moreg_i = (res[i * self.args_odd.spatial_dim :, :] - res[: -i * self.args_odd.spatial_dim, :]).norm(p=norm_ord)
                        moreg_i /= res[i * self.args_odd.spatial_dim :, :].norm(p=norm_ord) + res[: -i * self.args_odd.spatial_dim, :].norm(p=norm_ord)
                        moreg += moreg_i
                    moreg = moreg / len(self.args_odd.moreg_strides)
                    moreg = ((1 / self.args_odd.moreg_hyp[0] * moreg) ** self.args_odd.moreg_hyp[1]) / self.args_odd.moreg_hyp[2]
                else:
                    moreg = 1.0

                mograd = self.args_odd.mograd_mul * (moreg - self.args_odd.previous_moreg) / self.args_odd.skipped_step_length
                self.args_odd.previous_moreg = moreg
                moreg = moreg + abs(mograd)
                cache_diff = cache_diff * moreg

                metric_thres, cache_rates = list(self.codebook.keys()), list(self.codebook.values())
                if cache_diff < metric_thres[0]:
                    new_rate = cache_rates[0]
                elif cache_diff < metric_thres[1]:
                    new_rate = cache_rates[1]
                elif cache_diff < metric_thres[2]:
                    new_rate = cache_rates[2]
                elif cache_diff < metric_thres[3]:
                    new_rate = cache_rates[3]
                elif cache_diff < metric_thres[4]:
                    new_rate = cache_rates[4]
                else:
                    new_rate = cache_rates[-1]

                self.args_odd.previous_residual_tiny = self.args_odd.now_residual_tiny
                return new_rate

    def clear(self):
        if self.args_even.previous_residual is not None:
            self.args_even.previous_residual = self.args_even.previous_residual.cpu()
        if self.args_even.previous_residual_tiny is not None:
            self.args_even.previous_residual_tiny = self.args_even.previous_residual_tiny.cpu()
        if self.args_even.now_residual_tiny is not None:
            self.args_even.now_residual_tiny = self.args_even.now_residual_tiny.cpu()

        if self.args_odd.previous_residual is not None:
            self.args_odd.previous_residual = self.args_odd.previous_residual.cpu()
        if self.args_odd.previous_residual_tiny is not None:
            self.args_odd.previous_residual_tiny = self.args_odd.previous_residual_tiny.cpu()
        if self.args_odd.now_residual_tiny is not None:
            self.args_odd.now_residual_tiny = self.args_odd.now_residual_tiny.cpu()

        self.args_even.previous_residual = None
        self.args_even.previous_residual_tiny = None
        self.args_even.now_residual_tiny = None

        self.args_odd.previous_residual = None
        self.args_odd.previous_residual_tiny = None
        self.args_odd.now_residual_tiny = None

        torch.cuda.empty_cache()


class AdaArgs:
    def __init__(self, config):
        # Cache related attributes
        self.previous_residual_tiny = None
        self.now_residual_tiny = None
        self.norm_ord = 1
        self.skipped_step_length = 1
        self.previous_residual = None

        # Moreg related attributes
        self.previous_moreg = 1.0
        self.moreg_strides = [1]
        self.moreg_steps = [int(0.1 * config.infer_steps), int(0.9 * config.infer_steps)]
        self.moreg_hyp = [0.385, 8, 1, 2]
        self.mograd_mul = 10
        self.spatial_dim = 1536


class WanTransformerInferCustomCaching(WanTransformerInfer, BaseTaylorCachingTransformerInfer):
    def __init__(self, config):
        super().__init__(config)

        self.teacache_thresh = config.teacache_thresh
        self.accumulated_rel_l1_distance_even = 0
        self.previous_e0_even = None
        self.previous_residual_even = None
        self.accumulated_rel_l1_distance_odd = 0
        self.previous_e0_odd = None
        self.previous_residual_odd = None
        self.cache_even = {}
        self.cache_odd = {}

        self.use_ret_steps = config.use_ret_steps
        self.set_attributes_by_task_and_model()
        self.cnt = 0

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

    # 1. get taylor step_diff when there is two caching_records in scheduler
    def get_taylor_step_diff(self):
        step_diff = 0
        if self.infer_conditional:
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

    # calculate should_calc
    def calculate_should_calc(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        # 1. timestep embedding
        modulated_inp = embed0 if self.use_ret_steps else embed

        # 2. L1 calculate
        should_calc = False
        if self.infer_conditional:
            if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                should_calc = True
                self.accumulated_rel_l1_distance_even = 0
            else:
                rescale_func = np.poly1d(self.coefficients)
                self.accumulated_rel_l1_distance_even += rescale_func(((modulated_inp - self.previous_e0_even).abs().mean() / self.previous_e0_even.abs().mean()).cpu().item())
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
                self.accumulated_rel_l1_distance_odd += rescale_func(((modulated_inp - self.previous_e0_odd).abs().mean() / self.previous_e0_odd.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance_odd < self.teacache_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance_odd = 0
            self.previous_e0_odd = modulated_inp.clone()

        # 3. return the judgement
        return should_calc

    def infer(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        if self.infer_conditional:
            index = self.scheduler.step_index
            caching_records = self.scheduler.caching_records
            if index <= self.scheduler.infer_steps - 1:
                should_calc = self.calculate_should_calc(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)
                self.scheduler.caching_records[index] = should_calc

            if caching_records[index]:
                x = self.infer_calculating(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)
            else:
                x = self.infer_using_cache(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)

        else:
            index = self.scheduler.step_index
            caching_records_2 = self.scheduler.caching_records_2
            if index <= self.scheduler.infer_steps - 1:
                should_calc = self.calculate_should_calc(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)
                self.scheduler.caching_records_2[index] = should_calc

            if caching_records_2[index]:
                x = self.infer_calculating(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)
            else:
                x = self.infer_using_cache(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)

        if self.config.enable_cfg:
            self.switch_status()

        self.cnt += 1

        return x

    def infer_calculating(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        ori_x = x.clone()

        for block_idx in range(self.blocks_num):
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = self.infer_phase_1(weights.blocks[block_idx], grid_sizes, embed, x, embed0, seq_lens, freqs, context)

            y_out = self.infer_phase_2(weights.blocks[block_idx].compute_phases[0], grid_sizes, x, seq_lens, freqs, shift_msa, scale_msa)
            attn_out = self.infer_phase_3(weights.blocks[block_idx].compute_phases[1], x, context, y_out, gate_msa)
            y_out = self.infer_phase_4(weights.blocks[block_idx].compute_phases[2], x, attn_out, c_shift_msa, c_scale_msa)
            x = self.infer_phase_5(x, y_out, c_gate_msa)

        if self.infer_conditional:
            self.previous_residual_even = x - ori_x
            self.derivative_approximation(self.cache_even, "previous_residual", self.previous_residual_even)
        else:
            self.previous_residual_odd = x - ori_x
            self.derivative_approximation(self.cache_odd, "previous_residual", self.previous_residual_odd)
        return x

    def infer_using_cache(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        if self.infer_conditional:
            x += self.taylor_formula(self.cache_even["previous_residual"])
        else:
            x += self.taylor_formula(self.cache_odd["previous_residual"])
        return x

    def clear(self):
        if self.previous_residual_even is not None:
            self.previous_residual_even = self.previous_residual_even.cpu()
        if self.previous_residual_odd is not None:
            self.previous_residual_odd = self.previous_residual_odd.cpu()
        if self.previous_e0_even is not None:
            self.previous_e0_even = self.previous_e0_even.cpu()
        if self.previous_e0_odd is not None:
            self.previous_e0_odd = self.previous_e0_odd.cpu()

        for key in self.cache_even:
            if self.cache_even[key] is not None and hasattr(self.cache_even[key], "cpu"):
                self.cache_even[key] = self.cache_even[key].cpu()
        self.cache_even.clear()

        for key in self.cache_odd:
            if self.cache_odd[key] is not None and hasattr(self.cache_odd[key], "cpu"):
                self.cache_odd[key] = self.cache_odd[key].cpu()
        self.cache_odd.clear()

        self.previous_residual_even = None
        self.previous_residual_odd = None
        self.previous_e0_even = None
        self.previous_e0_odd = None

        torch.cuda.empty_cache()
