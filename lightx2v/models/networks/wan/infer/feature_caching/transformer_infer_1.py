from ..transformer_infer import BaseWanTransformer
import torch
from lightx2v.utils.envs import *


class WanTransformerInfer(BaseWanTransformer):
    def __init__(self, config):
        super().__init__(config)

    @torch.compile(disable=not CHECK_ENABLE_GRAPH_MODE())
    def infer(self, weights, grid_sizes, x, embed0, seq_lens, freqs, context):
        # 1. 读取数组
        index = self.scheduler.step_index
        caching_records = self.scheduler.caching_records

        # 2. 判断完全计算，或者使用缓存
        if caching_records[index]:
            return self.infer_calculating(weights, grid_sizes, x, embed0, seq_lens, freqs, context)
        else:
            return self.infer_using_cache(weights, grid_sizes, x, embed0, seq_lens, freqs, context)

    def infer_calculating(self, weights, grid_sizes, x, embed0, seq_lens, freqs, context):
        for block_idx in range(self.blocks_num):
            y_out, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = super().infer_block_1(weights, grid_sizes, x, embed0, seq_lens, freqs, context)
            attn_out = super().infer_block_2(weights, grid_sizes, x, embed0, seq_lens, freqs, context, y_out, gate_msa)
            y_out = super().infer_block_3(weights, grid_sizes, x, embed0, seq_lens, freqs, context, attn_out, c_shift_msa, c_scale_msa)
            x = super.infer_block_4(weights, grid_sizes, x, embed0, seq_lens, freqs, context, y_out, c_gate_msa)
        return x

    def infer_using_cache(self, weights, grid_sizes, x, embed0, seq_lens, freqs, context):
        pass