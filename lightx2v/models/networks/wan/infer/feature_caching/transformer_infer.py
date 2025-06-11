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
        super().set_attributes_by_task_and_model()
        self.cnt = 0

    def infer(self, weights, embed, grid_sizes, x, embed0, seq_lens, freqs, context):
        if self.infer_conditional:
            index = self.scheduler.step_index
            caching_records = self.scheduler.caching_records
            if index <= self.scheduler.infer_steps - 1:
                should_calc = super().calculate_should_calc(weights, embed, grid_sizes, x, embed0, seq_lens, freqs, context)
                self.scheduler.caching_records[index] = should_calc

            if caching_records[index]:
                x = self.infer_calculating(weights, grid_sizes, x, embed0, seq_lens, freqs, context)
            else:
                x = self.infer_using_cache(weights, grid_sizes, x, embed0, seq_lens, freqs, context)

        else:
            index = self.scheduler.step_index
            caching_records_2 = self.scheduler.caching_records_2
            if index <= self.scheduler.infer_steps - 1:
                should_calc = super().calculate_should_calc(weights, embed, grid_sizes, x, embed0, seq_lens, freqs, context)
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


class WanTransformerInferAdaCaching(BaseWanTransformerInfer):
    def __init__(self, config):
        super().__init__(config)

        # 1. fixed args
        self.decisive_double_block_id = self.blocks_num // 2
        self.codebook = {0.03: 12, 0.05: 10, 0.07: 8, 0.09: 6, 0.11: 4, 1.00: 3}
        
        # 2. Create two instances of AdaArgs
        self.args_even = AdaArgs(config)
        self.args_odd = AdaArgs(config)

    def infer(self, weights, embed, grid_sizes, x, embed0, seq_lens, freqs, context):
        if self.infer_conditional:
            index = self.scheduler.step_index
            caching_records = self.scheduler.caching_records

            if caching_records[index]:
                x = self.infer_calculating(weights, grid_sizes, x, embed0, seq_lens, freqs, context)

                # 1. 计算接下来需要跳过的步数
                if index <= self.scheduler.infer_steps - 2:
                    self.args_even.skipped_step_length = self.calculate_skip_step_length()
                    for i in range(1, self.args_even.skipped_step_length):
                        if (index + i) <= self.scheduler.infer_steps - 1:
                            self.scheduler.caching_records[index+i] = False
            else:
                x = self.infer_using_cache(weights, grid_sizes, x, embed0, seq_lens, freqs, context)

        else:
            index = self.scheduler.step_index
            caching_records = self.scheduler.caching_records_2

            if caching_records[index]:
                x = self.infer_calculating(weights, grid_sizes, x, embed0, seq_lens, freqs, context)

                # 1. 计算接下来需要跳过的步数
                if index <= self.scheduler.infer_steps - 2:
                    self.args_odd.skipped_step_length = self.calculate_skip_step_length()
                    for i in range(1, self.args_odd.skipped_step_length):
                        if (index + i) <= self.scheduler.infer_steps - 1:
                            self.scheduler.caching_records_2[index+i] = False
            else:
                x = self.infer_using_cache(weights, grid_sizes, x, embed0, seq_lens, freqs, context)

        if self.config.enable_cfg:
            super().switch_status()

        return x
        

    def infer_calculating(self, weights, grid_sizes, x, embed0, seq_lens, freqs, context):
        ori_x = x.clone()

        for block_idx in range(self.blocks_num):
            y_out, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = super().infer_block_1(weights.blocks[block_idx], grid_sizes, x, embed0, seq_lens, freqs, context)
            if block_idx == self.decisive_double_block_id:
                if self.infer_conditional:
                    self.args_even.now_residual_tiny = y_out * gate_msa.squeeze(0)
                else:
                    self.args_odd.now_residual_tiny = y_out * gate_msa.squeeze(0)

            attn_out = super().infer_block_2(weights.blocks[block_idx], grid_sizes, x, embed0, seq_lens, freqs, context, y_out, gate_msa)
            y_out = super().infer_block_3(weights.blocks[block_idx], grid_sizes, x, embed0, seq_lens, freqs, context, attn_out, c_shift_msa, c_scale_msa)
            x = super().infer_block_4(weights.blocks[block_idx], grid_sizes, x, embed0, seq_lens, freqs, context, y_out, c_gate_msa)

        if self.infer_conditional:
            self.args_even.previous_residual = x - ori_x
        else:
            self.args_odd.previous_residual = x - ori_x
        return x
    
    def infer_using_cache(self, weights, grid_sizes, x, embed0, seq_lens, freqs, context):
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
                cache_diff = (cache - res).norm(dim=(0,1), p=norm_ord) / cache.norm(dim=(0,1), p=norm_ord)
                cache_diff = cache_diff / self.args_even.skipped_step_length

                if self.args_even.moreg_steps[0] <= self.scheduler.step_index <= self.args_even.moreg_steps[1]:
                    moreg = 0
                    for i in self.args_even.moreg_strides:
                        moreg_i = (res[i*self.args_even.spatial_dim:, :] - res[:-i*self.args_even.spatial_dim, :]).norm(p=norm_ord) 
                        moreg_i /= (res[i*self.args_even.spatial_dim:, :].norm(p=norm_ord) + res[:-i*self.args_even.spatial_dim, :].norm(p=norm_ord))
                        moreg += moreg_i
                    moreg = moreg / len(self.args_even.moreg_strides)
                    moreg = ((1/self.args_even.moreg_hyp[0] * moreg) ** self.args_even.moreg_hyp[1]) / self.args_even.moreg_hyp[2] 
                else:
                    moreg = 1.

                mograd = self.args_even.mograd_mul * (moreg - self.args_even.previous_moreg) / self.args_even.skipped_step_length
                self.args_even.previous_moreg = moreg
                moreg = moreg + abs(mograd)
                cache_diff = cache_diff * moreg

                metric_thres, cache_rates = list(self.codebook.keys()), list(self.codebook.values())
                if cache_diff < metric_thres[0]: new_rate = cache_rates[0]
                elif cache_diff < metric_thres[1]: new_rate = cache_rates[1]
                elif cache_diff < metric_thres[2]: new_rate = cache_rates[2]
                elif cache_diff < metric_thres[3]: new_rate = cache_rates[3]
                elif cache_diff < metric_thres[4]: new_rate = cache_rates[4]
                else: new_rate = cache_rates[-1]
                
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
                cache_diff = (cache - res).norm(dim=(0,1), p=norm_ord) / cache.norm(dim=(0,1), p=norm_ord)
                cache_diff = cache_diff / self.args_odd.skipped_step_length

                if self.args_odd.moreg_steps[0] <= self.scheduler.step_index <= self.args_odd.moreg_steps[1]:
                    moreg = 0
                    for i in self.args_odd.moreg_strides:
                        moreg_i = (res[i*self.args_odd.spatial_dim:, :] - res[:-i*self.args_odd.spatial_dim, :]).norm(p=norm_ord) 
                        moreg_i /= (res[i*self.args_odd.spatial_dim:, :].norm(p=norm_ord) + res[:-i*self.args_odd.spatial_dim, :].norm(p=norm_ord))
                        moreg += moreg_i
                    moreg = moreg / len(self.args_odd.moreg_strides)
                    moreg = ((1/self.args_odd.moreg_hyp[0] * moreg) ** self.args_odd.moreg_hyp[1]) / self.args_odd.moreg_hyp[2] 
                else:
                    moreg = 1.

                mograd = self.args_odd.mograd_mul * (moreg - self.args_odd.previous_moreg) / self.args_odd.skipped_step_length
                self.args_odd.previous_moreg = moreg
                moreg = moreg + abs(mograd)
                cache_diff = cache_diff * moreg

                metric_thres, cache_rates = list(self.codebook.keys()), list(self.codebook.values())
                if cache_diff < metric_thres[0]: new_rate = cache_rates[0]
                elif cache_diff < metric_thres[1]: new_rate = cache_rates[1]
                elif cache_diff < metric_thres[2]: new_rate = cache_rates[2]
                elif cache_diff < metric_thres[3]: new_rate = cache_rates[3]
                elif cache_diff < metric_thres[4]: new_rate = cache_rates[4]
                else: new_rate = cache_rates[-1]
                
                self.args_odd.previous_residual_tiny = self.args_odd.now_residual_tiny
                return new_rate


class AdaArgs:
    def __init__(self, config):
        # Cache related attributes
        self.previous_residual_tiny = None
        self.now_residual_tiny = None
        self.norm_ord = 1
        self.skipped_step_length = 1
        self.previous_residual = None

        # Moreg related attributes  
        self.previous_moreg = 1.
        self.moreg_strides = [1]
        self.moreg_steps = [
            int(0.1 * config.infer_steps),
            int(0.9 * config.infer_steps)
        ]
        self.moreg_hyp = [0.385, 8, 1, 2]
        self.mograd_mul = 10
        self.spatial_dim = 1536


class WanTransformerInferCustomCaching(BaseWanTransformerInfer):
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
        super().set_attributes_by_task_and_model()
        self.cnt = 0

    def infer(self, weights, embed, grid_sizes, x, embed0, seq_lens, freqs, context):
        if self.infer_conditional:
            index = self.scheduler.step_index
            caching_records = self.scheduler.caching_records
            if index <= self.scheduler.infer_steps - 1:
                should_calc = super().calculate_should_calc(weights, embed, grid_sizes, x, embed0, seq_lens, freqs, context)
                self.scheduler.caching_records[index] = should_calc

            if caching_records[index]:
                x = self.infer_calculating(weights, grid_sizes, x, embed0, seq_lens, freqs, context)
            else:
                x = self.infer_using_cache(weights, grid_sizes, x, embed0, seq_lens, freqs, context)

        else:
            index = self.scheduler.step_index
            caching_records_2 = self.scheduler.caching_records_2
            if index <= self.scheduler.infer_steps - 1:
                should_calc = super().calculate_should_calc(weights, embed, grid_sizes, x, embed0, seq_lens, freqs, context)
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
            super.derivative_approximation(self.cache_even, "previous_residual", self.previous_residual_even)
        else:
            self.previous_residual_odd = x - ori_x
            super.derivative_approximation(self.cache_odd, "previous_residual", self.previous_residual_odd)
        return x

    def infer_using_cache(self, weights, grid_sizes, x, embed0, seq_lens, freqs, context):
        if self.infer_conditional:
            x += super().taylor_formula(self.cache_even["previous_residual"])
        else:
            x += super().taylor_formula(self.cache_odd["previous_residual"])
        return x