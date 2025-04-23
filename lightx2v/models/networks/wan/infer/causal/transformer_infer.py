import torch
import math
from ..utils import compute_freqs, compute_freqs_causal, compute_freqs_dist, apply_rotary_emb
from lightx2v.attentions import attention
from lightx2v.common.offload.manager import WeightStreamManager
from lightx2v.utils.envs import *
from ..transformer_infer import WanTransformerInfer


class WanTransformerInferCausal(WanTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self.num_frames = config["num_frames"]
        self.num_frame_per_block = config["num_frame_per_block"]
        self.frame_seq_length = config["frame_seq_length"]
        self.text_len = config["text_len"]
        self.kv_size = self.num_frames * self.frame_seq_length
        self.kv_cache = None
        self.crossattn_cache = None

    def _init_kv_cache(self, dtype, device):
        kv_cache = []

        for _ in range(self.blocks_num):
            kv_cache.append({
                "k": torch.zeros([self.kv_size, self.num_heads, self.head_dim], dtype=dtype, device=device),
                "v": torch.zeros([self.kv_size, self.num_heads, self.head_dim], dtype=dtype, device=device)
            })

        self.kv_cache = kv_cache

    def _init_crossattn_cache(self, dtype, device):
        crossattn_cache = []

        for _ in range(self.blocks_num):
            crossattn_cache.append({
                "k": torch.zeros([self.text_len, self.num_heads, self.head_dim], dtype=dtype, device=device),
                "v": torch.zeros([self.text_len, self.num_heads, self.head_dim], dtype=dtype, device=device),
                "is_init": False
            })

        self.crossattn_cache = crossattn_cache

    @torch.compile(disable=not CHECK_ENABLE_GRAPH_MODE())
    def infer(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context, kv_start, kv_end):
        return self.infer_func(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context, kv_start, kv_end)

    def _infer_with_offload(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context, kv_start, kv_end):
        for block_idx in range(self.blocks_num):
            if block_idx == 0:
                self.weights_stream_mgr.active_weights[0] = weights.blocks_weights[0]
                self.weights_stream_mgr.active_weights[0].to_cuda()

            with torch.cuda.stream(self.weights_stream_mgr.compute_stream):
                x = self.infer_block(
                    self.weights_stream_mgr.active_weights[0],
                    grid_sizes,
                    embed,
                    x,
                    embed0,
                    seq_lens,
                    freqs,
                    context,
                    block_idx,
                    kv_start,
                    kv_end
                )

            if block_idx < self.blocks_num - 1:
                self.weights_stream_mgr.prefetch_weights(block_idx + 1, weights.blocks_weights)
            self.weights_stream_mgr.swap_weights()

        return x

    def _infer_without_offload(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context, kv_start, kv_end):
        for block_idx in range(self.blocks_num):
            x = self.infer_block(
                weights.blocks_weights[block_idx],
                grid_sizes,
                embed,
                x,
                embed0,
                seq_lens,
                freqs,
                context,
                block_idx,
                kv_start,
                kv_end
            )
        return x
    
    def infer_block(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context, block_idx, kv_start, kv_end):
        embed0 = (weights.modulation + embed0).chunk(6, dim=1)

        norm1_out = torch.nn.functional.layer_norm(x, (x.shape[1],), None, None, 1e-6)
        norm1_out = (norm1_out * (1 + embed0[1]) + embed0[0]).squeeze(0)

        s, n, d = *norm1_out.shape[:1], self.num_heads, self.head_dim
        q = weights.self_attn_norm_q.apply(weights.self_attn_q.apply(norm1_out)).view(s, n, d)
        k = weights.self_attn_norm_k.apply(weights.self_attn_k.apply(norm1_out)).view(s, n, d)
        v = weights.self_attn_v.apply(norm1_out).view(s, n, d)

        if not self.parallel_attention:
            freqs_i = compute_freqs_causal(
                q.size(2) // 2,
                grid_sizes,
                freqs,
                start_frame=kv_start // math.prod(grid_sizes[0][1:]).item()
            )
        else:
            # TODO: Implement parallel attention for causal inference
            raise NotImplementedError("Parallel attention is not implemented for causal inference")
        
        q = apply_rotary_emb(q, freqs_i)
        k = apply_rotary_emb(k, freqs_i)

        self.kv_cache[block_idx]["k"][kv_start:kv_end] = k
        self.kv_cache[block_idx]["v"][kv_start:kv_end] = v

        cu_seqlens_q, cu_seqlens_k, lq, lk = self._calculate_q_k_len(
            q = q, 
            k = self.kv_cache[block_idx]["k"][:kv_end], 
            k_lens=torch.tensor([kv_end], dtype=torch.int32, device=k.device)
        )

        if not self.parallel_attention:
            attn_out = attention(
                attention_type = self.attention_type, 
                q = q, 
                k = self.kv_cache[block_idx]["k"][:kv_end], 
                v = self.kv_cache[block_idx]["v"][:kv_end], 
                cu_seqlens_q = cu_seqlens_q, 
                cu_seqlens_kv = cu_seqlens_k, 
                max_seqlen_q = lq, 
                max_seqlen_kv = lk, 
                model_cls = self.config["model_cls"]
            )
        else:
            # TODO: Implement parallel attention for causal inference
            raise NotImplementedError("Parallel attention is not implemented for causal inference")
        
        y = weights.self_attn_o.apply(attn_out)

        x = x + y * embed0[2].squeeze(0)

        norm3_out = weights.norm3.apply(x)

        # TODO: Implement I2V inference for causal model
        if self.task == "i2v":
            raise NotImplementedError("I2V inference for causal model is not implemented")

        n, d = self.num_heads, self.head_dim
        q = weights.cross_attn_norm_q.apply(weights.cross_attn_q.apply(norm3_out)).view(-1, n, d)
        if not self.crossattn_cache[block_idx]["is_init"]:
            k = weights.cross_attn_norm_k.apply(weights.cross_attn_k.apply(context)).view(-1, n, d)
            v = weights.cross_attn_v.apply(context).view(-1, n, d)
            self.crossattn_cache[block_idx]["k"] = k
            self.crossattn_cache[block_idx]["v"] = v
            self.crossattn_cache[block_idx]["is_init"] = True
        else:
            k = self.crossattn_cache[block_idx]["k"]
            v = self.crossattn_cache[block_idx]["v"]

        cu_seqlens_q, cu_seqlens_k, lq, lk = self._calculate_q_k_len(q, k, k_lens=torch.tensor([k.size(0)], dtype=torch.int32, device=k.device))

        attn_out = attention(
            attention_type=self.attention_type, q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_k, max_seqlen_q=lq, max_seqlen_kv=lk, model_cls=self.config["model_cls"]
        )

        # TODO: Implement I2V inference for causal model
        if self.task == "i2v":
            raise NotImplementedError("I2V inference for causal model is not implemented")
        
        attn_out = weights.cross_attn_o.apply(attn_out)

        x = x + attn_out
        norm2_out = torch.nn.functional.layer_norm(x, (x.shape[1],), None, None, 1e-6)
        y = weights.ffn_0.apply(norm2_out * (1 + embed0[4].squeeze(0)) + embed0[3].squeeze(0))
        y = torch.nn.functional.gelu(y, approximate="tanh")
        y = weights.ffn_2.apply(y)
        x = x + y * embed0[5].squeeze(0)
        return x
        