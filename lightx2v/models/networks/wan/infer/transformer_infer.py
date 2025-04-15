import torch
from .utils import compute_freqs, compute_freqs_dist, apply_rotary_emb
from lightx2v.attentions import attention
from lightx2v.common.offload.manager import WeightStreamManager
from lightx2v.utils.envs import *


class WanTransformerInfer:
    def __init__(self, config):
        self.config = config
        self.task = config["task"]
        self.attention_type = config.get("attention_type", "flash_attn2")
        self.blocks_num = config["num_layers"]
        self.num_heads = config["num_heads"]
        self.head_dim = config["dim"] // config["num_heads"]
        self.window_size = config.get("window_size", (-1, -1))
        self.parallel_attention = None
        if self.config["cpu_offload"]:
            self.weights_stream_mgr = WeightStreamManager()
            self.infer_func = self._infer_with_offload
        else:
            self.infer_func = self._infer_without_offload

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def _calculate_q_k_len(self, q, k, k_lens):
        lq, nq, c1 = q.size()
        lk, nk, c1_k = k.size()
        # Handle query and key lengths (use `q_lens` and `k_lens` or set them to Lq and Lk if None)
        q_lens = torch.tensor([lq], dtype=torch.int32, device=q.device)

        # We don't have a batch dimension anymore, so directly use the `q_lens` and `k_lens` values
        cu_seqlens_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32)
        cu_seqlens_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32)
        return cu_seqlens_q, cu_seqlens_k, lq, lk

    @torch.compile(disable=not CHECK_ENABLE_GRAPH_MODE())
    def infer(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        return self.infer_func(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)

    def _infer_with_offload(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        for block_idx in range(self.blocks_num):
            if block_idx == 0:
                self.weights_stream_mgr.active_weights[0] = weights.blocks[0]
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
                )

            if block_idx < self.blocks_num - 1:
                self.weights_stream_mgr.prefetch_weights(block_idx + 1, weights.blocks)
            self.weights_stream_mgr.swap_weights()

        return x

    def _infer_without_offload(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        for block_idx in range(self.blocks_num):
            x = self.infer_block(
                weights.blocks[block_idx],
                grid_sizes,
                embed,
                x,
                embed0,
                seq_lens,
                freqs,
                context,
            )
        return x

    def infer_block(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        if embed0.dim() == 3:
            modulation = weights.modulation.tensor.unsqueeze(2)  # 1, 6, 1, dim
            embed0 = embed0.unsqueeze(0)  #
            embed0 = (modulation + embed0).chunk(6, dim=1)
            embed0 = [ei.squeeze(1) for ei in embed0]
        elif embed0.dim() == 2:
            embed0 = (weights.modulation.tensor + embed0).chunk(6, dim=1)

        norm1_out = torch.nn.functional.layer_norm(x, (x.shape[1],), None, None, 1e-6)
        norm1_out = (norm1_out * (1 + embed0[1]) + embed0[0]).squeeze(0)

        s, n, d = *norm1_out.shape[:1], self.num_heads, self.head_dim
        q = weights.self_attn_norm_q.apply(weights.self_attn_q.apply(norm1_out)).view(s, n, d)
        k = weights.self_attn_norm_k.apply(weights.self_attn_k.apply(norm1_out)).view(s, n, d)
        v = weights.self_attn_v.apply(norm1_out).view(s, n, d)

        if not self.parallel_attention:
            freqs_i = compute_freqs(q.size(2) // 2, grid_sizes, freqs)
        else:
            freqs_i = compute_freqs_dist(q.size(0), q.size(2) // 2, grid_sizes, freqs)

        q = apply_rotary_emb(q, freqs_i)
        k = apply_rotary_emb(k, freqs_i)

        cu_seqlens_q, cu_seqlens_k, lq, lk = self._calculate_q_k_len(q, k, k_lens=seq_lens)

        if not self.parallel_attention:
            attn_out = weights.self_attn_1.apply(q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_k, max_seqlen_q=lq, max_seqlen_kv=lk, model_cls=self.config["model_cls"])
        else:
            attn_out = self.parallel_attention(
                attention_type=self.attention_type,
                q=q,
                k=k,
                v=v,
                img_qkv_len=q.shape[0],
                cu_seqlens_qkv=cu_seqlens_q,
                # cu_seqlens_qkv=cu_seqlens_qkv,
                # max_seqlen_qkv=max_seqlen_qkv,
            )
        y = weights.self_attn_o.apply(attn_out)

        x = x + y * embed0[2].squeeze(0)

        norm3_out = weights.norm3.apply(x)

        if self.task == "i2v":
            context_img = context[:257]
            context = context[257:]

        n, d = self.num_heads, self.head_dim
        q = weights.cross_attn_norm_q.apply(weights.cross_attn_q.apply(norm3_out)).view(-1, n, d)
        k = weights.cross_attn_norm_k.apply(weights.cross_attn_k.apply(context)).view(-1, n, d)
        v = weights.cross_attn_v.apply(context).view(-1, n, d)

        cu_seqlens_q, cu_seqlens_k, lq, lk = self._calculate_q_k_len(q, k, k_lens=torch.tensor([k.size(0)], dtype=torch.int32, device=k.device))

        attn_out = weights.cross_attn_1.apply(q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_k, max_seqlen_q=lq, max_seqlen_kv=lk, model_cls=self.config["model_cls"])

        if self.task == "i2v":
            k_img = weights.cross_attn_norm_k_img.apply(weights.cross_attn_k_img.apply(context_img)).view(-1, n, d)
            v_img = weights.cross_attn_v_img.apply(context_img).view(-1, n, d)

            cu_seqlens_q, cu_seqlens_k, lq, lk = self._calculate_q_k_len(
                q,
                k_img,
                k_lens=torch.tensor([k_img.size(0)], dtype=torch.int32, device=k.device),
            )

            img_attn_out = weights.cross_attn_2.apply(
                attention_type=self.attention_type, q=q, k=k_img, v=v_img, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_k, max_seqlen_q=lq, max_seqlen_kv=lk, model_cls=self.config["model_cls"]
            )

            attn_out = attn_out + img_attn_out

        attn_out = weights.cross_attn_o.apply(attn_out)

        x = x + attn_out
        norm2_out = torch.nn.functional.layer_norm(x, (x.shape[1],), None, None, 1e-6)
        y = weights.ffn_0.apply(norm2_out * (1 + embed0[4].squeeze(0)) + embed0[3].squeeze(0))
        y = torch.nn.functional.gelu(y, approximate="tanh")
        y = weights.ffn_2.apply(y)
        x = x + y * embed0[5].squeeze(0)
        return x
