from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
import torch
from .utils import compute_freqs, compute_freqs_dist, apply_rotary_emb
from lightx2v.common.offload.manager import WeightAsyncStreamManager
from lightx2v.utils.envs import *


class BaseWanTransformerInfer(BaseTransformerInfer):
    # 1. 初始化
    def __init__(self, config):
        # 1.1 配置 + 任务 + 注意力
        self.config = config
        self.task = config["task"]
        self.attention_type = config.get("attention_type", "flash_attn2")

        # 1.2 块数量:30, 头数量, 输入维度, 窗口大小
        self.blocks_num = config["num_layers"]
        self.num_heads = config["num_heads"]
        self.head_dim = config["dim"] // config["num_heads"]
        self.window_size = config.get("window_size", (-1, -1))
        self.parallel_attention = None

        # 1.3 缓存切换状态专用
        self.infer_conditional = True

    # per block
    def infer_block_1(self, weights, grid_sizes, x, embed0, seq_lens, freqs, context):
        # 1. pass to all the part
        if embed0.dim() == 3:
            modulation = weights.modulation.tensor.unsqueeze(2)
            embed0 = (modulation + embed0).chunk(6, dim=1)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = [ei.squeeze(1) for ei in embed0]
        elif embed0.dim() == 2:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (weights.modulation.tensor + embed0).chunk(6, dim=1)

        # 2. first part before +
        if hasattr(weights.compute_phases[0], "smooth_norm1_weight"):
            norm1_weight = (1 + scale_msa) * weights.compute_phases[0].smooth_norm1_weight.tensor
            norm1_bias = shift_msa * weights.compute_phases[0].smooth_norm1_bias.tensor
        else:
            norm1_weight = 1 + scale_msa
            norm1_bias = shift_msa

        norm1_out = torch.nn.functional.layer_norm(x, (x.shape[1],), None, None, 1e-6)
        norm1_out = (norm1_out * norm1_weight + norm1_bias).squeeze(0)

        s, n, d = *norm1_out.shape[:1], self.num_heads, self.head_dim
        q = weights.compute_phases[0].self_attn_norm_q.apply(weights.compute_phases[0].self_attn_q.apply(norm1_out)).view(s, n, d)
        k = weights.compute_phases[0].self_attn_norm_k.apply(weights.compute_phases[0].self_attn_k.apply(norm1_out)).view(s, n, d)
        v = weights.compute_phases[0].self_attn_v.apply(norm1_out).view(s, n, d)

        if not self.parallel_attention:
            freqs_i = compute_freqs(q.size(2) // 2, grid_sizes, freqs)
        else:
            freqs_i = compute_freqs_dist(q.size(0), q.size(2) // 2, grid_sizes, freqs)

        q = apply_rotary_emb(q, freqs_i)
        k = apply_rotary_emb(k, freqs_i)

        cu_seqlens_q, cu_seqlens_k = self._calculate_q_k_len(q, k_lens=seq_lens)

        if not self.parallel_attention:
            attn_out = weights.compute_phases[0].self_attn_1.apply(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_k,
                max_seqlen_q=q.size(0),
                max_seqlen_kv=k.size(0),
                model_cls=self.config["model_cls"],
            )
        else:
            attn_out = self.parallel_attention(
                attention_type=self.attention_type,
                q=q,
                k=k,
                v=v,
                img_qkv_len=q.shape[0],
                cu_seqlens_qkv=cu_seqlens_q,
            )

        y = weights.compute_phases[0].self_attn_o.apply(attn_out)
        return y, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa

    def infer_block_2(self, weights, grid_sizes, x, embed0, seq_lens, freqs, context, y_out, gate_msa):
        x.add_(y_out * gate_msa.squeeze(0))

        norm3_out = weights.compute_phases[1].norm3.apply(x)

        if self.task == "i2v":
            context_img = context[:257]
            context = context[257:]
        else:
            context_img = None

        n, d = self.num_heads, self.head_dim
        q = weights.compute_phases[1].cross_attn_norm_q.apply(weights.compute_phases[1].cross_attn_q.apply(norm3_out)).view(-1, n, d)
        k = weights.compute_phases[1].cross_attn_norm_k.apply(weights.compute_phases[1].cross_attn_k.apply(context)).view(-1, n, d)
        v = weights.compute_phases[1].cross_attn_v.apply(context).view(-1, n, d)

        cu_seqlens_q, cu_seqlens_k = self._calculate_q_k_len(
            q,
            k_lens=torch.tensor([k.size(0)], dtype=torch.int32, device=k.device),
        )

        attn_out = weights.compute_phases[1].cross_attn_1.apply(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_k,
            max_seqlen_q=q.size(0),
            max_seqlen_kv=k.size(0),
            model_cls=self.config["model_cls"],
        )

        if self.task == "i2v" and context_img is not None:
            k_img = weights.compute_phases[1].cross_attn_norm_k_img.apply(weights.compute_phases[1].cross_attn_k_img.apply(context_img)).view(-1, n, d)
            v_img = weights.compute_phases[1].cross_attn_v_img.apply(context_img).view(-1, n, d)

            cu_seqlens_q, cu_seqlens_k = self._calculate_q_k_len(
                q,
                k_lens=torch.tensor([k_img.size(0)], dtype=torch.int32, device=k.device),
            )

            img_attn_out = weights.compute_phases[1].cross_attn_2.apply(
                q=q,
                k=k_img,
                v=v_img,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_k,
                max_seqlen_q=q.size(0),
                max_seqlen_kv=k_img.size(0),
                model_cls=self.config["model_cls"],
            )

            attn_out = attn_out + img_attn_out

        attn_out = weights.compute_phases[1].cross_attn_o.apply(attn_out)
        return attn_out

    def infer_block_3(self, weights, grid_sizes, x, embed0, seq_lens, freqs, context, attn_out, c_shift_msa, c_scale_msa):
        x.add_(attn_out)

        if hasattr(weights.compute_phases[2], "smooth_norm2_weight"):
            norm2_weight = (1 + c_scale_msa.squeeze(0)) * weights.compute_phases[2].smooth_norm2_weight.tensor
            norm2_bias = c_shift_msa.squeeze(0) * weights.compute_phases[2].smooth_norm2_bias.tensor
        else:
            norm2_weight = 1 + c_scale_msa.squeeze(0)
            norm2_bias = c_shift_msa.squeeze(0)

        norm2_out = torch.nn.functional.layer_norm(x, (x.shape[1],), None, None, 1e-6)
        y = weights.compute_phases[2].ffn_0.apply(norm2_out * norm2_weight + norm2_bias)
        y = torch.nn.functional.gelu(y, approximate="tanh")
        y = weights.compute_phases[2].ffn_2.apply(y)
        return y

    def infer_block_4(self, weights, grid_sizes, x, embed0, seq_lens, freqs, context, y_out, c_gate_msa):
        x.add_(y_out * c_gate_msa.squeeze(0))
        return x

    def _calculate_q_k_len(self, q, k_lens):
        # Handle query and key lengths (use `q_lens` and `k_lens` or set them to Lq and Lk if None)
        q_lens = torch.tensor([q.size(0)], dtype=torch.int32, device=q.device)

        # We don't have a batch dimension anymore, so directly use the `q_lens` and `k_lens` values
        cu_seqlens_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32)
        cu_seqlens_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32)
        return cu_seqlens_q, cu_seqlens_k
    
    def switch_status(self):
        self.infer_conditional = not self.infer_conditional

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


class WanTransformerInfer(BaseWanTransformerInfer):
    def __init__(self, config):
        super().__init__(config)

    @torch.compile(disable=not CHECK_ENABLE_GRAPH_MODE())
    def infer(self, weights, embed, grid_sizes, x, embed0, seq_lens, freqs, context):
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
            y_out, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = super().infer_block_1(weights.blocks[block_idx], grid_sizes, x, embed0, seq_lens, freqs, context)
            attn_out = super().infer_block_2(weights.blocks[block_idx], grid_sizes, x, embed0, seq_lens, freqs, context, y_out, gate_msa)
            y_out = super().infer_block_3(weights.blocks[block_idx], grid_sizes, x, embed0, seq_lens, freqs, context, attn_out, c_shift_msa, c_scale_msa)
            x = super().infer_block_4(weights.blocks[block_idx], grid_sizes, x, embed0, seq_lens, freqs, context, y_out, c_gate_msa)
        return x

    def infer_using_cache(self, weights, grid_sizes, x, embed0, seq_lens, freqs, context):
        pass