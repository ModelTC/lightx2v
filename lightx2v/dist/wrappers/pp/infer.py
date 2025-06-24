import enum
from requests import patch
import torch
import torch.distributed as dist
import torch.nn.functional as F
from lightx2v.utils.envs import *
from lightx2v.models.networks.wan.infer.transformer_infer import (
    WanTransformerInfer,
)
from lightx2v.models.networks.wan.infer.utils import compute_freqs, compute_freqs_dist, compute_freqs_by_patch, apply_rotary_emb
from lightx2v.dist.wrappers.pp.kv_cache import PipelineParallelKVCacheManager, RingAttnCacheManager
from lightx2v.attentions.distributed.ring.attn import ring_attn_sub, update_out_and_lse
from loguru import logger


class PipelineParallelWanTransformerInferWrapper:
    def __init__(self, transformer_infer: WanTransformerInfer, config):
        self.transformer_infer = transformer_infer
        self.config = config
        
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        self.patch_num=self.config.get("patch_num", 2)
        self.patch_results = [None for i in range(self.patch_num)]
        
        self.reset_block_index()
        
        self.infer_func = self._infer_without_offload
        
        self.use_kv_cache = True
        self.use_ring_attn_cache = False
        
        
    def set_blocks_num(self, blocks_num):
        self.transformer_infer.blocks_num = blocks_num
        
    def reset_block_index(self):
        self.block_index = 0
        
    def init_kv_cache_manager(self):
        self.kv_cache_manager = PipelineParallelKVCacheManager(kv_cache_len=self.transformer_infer.blocks_num, patch_num=self.patch_num)
        self.ring_attn_cache_manager = RingAttnCacheManager(cache_len=self.transformer_infer.blocks_num, patch_num=self.patch_num)
        
    def get_patch_inputs(self, x):
        patch_size = self.patch_num
        padding_size = (patch_size - (x.shape[0] % patch_size)) % patch_size
        if padding_size > 0:
            # 使用 F.pad 填充第一维
            x = F.pad(x, (0, 0, 0, padding_size))  # (后维度填充, 前维度填充)
        xs = torch.chunk(x, patch_size, dim=0)
        return xs
    
    def get_full_output(self, xs):
        x = torch.cat(xs, dim=0)
        return x
        
    def update_kv_cache_manager(self, full_key, full_value):
        keys = torch.chunk(full_key, self.patch_num, dim=0)
        values = torch.chunk(full_value, self.patch_num, dim=0)
        for index, (k, v) in enumerate(zip(keys, values)):
            self.kv_cache_manager.update_kv_cache(k, v, index, self.block_index)
            
    def update_kv_cache_manager_by_patch_index(self, key, value):
        self.kv_cache_manager.update_kv_cache(key, value, self.patch_index, self.block_index)
    
    def get_kv_cache_from_manager(self, cur_key, cur_value):
        return self.kv_cache_manager.get_full_kv_with_cache(cur_key, cur_value, self.patch_index, self.block_index)

    def __getattr__(self, name: str):
        if name in self.__dict__:
            return getattr(self, name)
        else:
            return getattr(self.transformer_infer, name)
        
    def _infer_without_offload(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        for block_idx in range(self.blocks_num):
            self.block_index = block_idx
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
        
    # @torch.compile(disable=not CHECK_ENABLE_GRAPH_MODE())
    def infer(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context, is_warmup):
        if is_warmup:
            # logger.info(f"warmup infer is running")
            self.transformer_infer._infer_self_attn = self._infer_self_attn
            if self.rank == 0:
                x = self.infer_func(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)
                dist.send(x, dst=self.rank+1)
            elif self.rank == self.world_size-1:
                dist.recv(x, src=self.rank-1)
                x = self.infer_func(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)
            else:
                dist.recv(x, src=self.rank-1)
                x = self.infer_func(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)
                dist.send(x, dst=self.rank+1)
            dist.broadcast(x, src=self.world_size-1)
        else:
            # logger.info(f"kv-cache infer is running")
            self.transformer_infer._infer_self_attn = self._infer_self_attn_cached
            ori_x = x
            xs = self.get_patch_inputs(x)
            # if self.rank == 0:
            #     import pdb; pdb.set_trace()
            for patch_index, x in enumerate(xs):
                self.patch_index = patch_index
                if self.rank == 0:
                    x = self.infer_func(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)
                    dist.send(x, dst=self.rank+1)
                elif self.rank == self.world_size-1:
                    dist.recv(x, src=self.rank-1)
                    x = self.infer_func(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)
                    self.patch_results[patch_index] = x
                else:
                    dist.recv(x, src=self.rank-1)
                    x = self.infer_func(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)
                    dist.send(x, dst=self.rank+1)
            if self.rank == self.world_size-1:
                full_x = torch.cat(self.patch_results, dim=0)
            else:
                full_x = torch.empty_like(ori_x)
            dist.broadcast(full_x, src=self.world_size-1)
            x = full_x
        return x
    
    def _infer_self_attn_cached(self, weights, x, shift_msa, scale_msa, gate_msa, grid_sizes, freqs, seq_lens):
        # logger.info(f"[RANK{self.rank}] cur block idx: {self.block_index}, cur patch idx: {self.patch_index}")
        if hasattr(weights, "smooth_norm1_weight"):
            norm1_weight = (1 + scale_msa) * weights.smooth_norm1_weight.tensor
            norm1_bias = shift_msa * weights.smooth_norm1_bias.tensor
        else:
            norm1_weight = 1 + scale_msa
            norm1_bias = shift_msa

        norm1_out = torch.nn.functional.layer_norm(x, (x.shape[1],), None, None, 1e-6)
        norm1_out = (norm1_out * norm1_weight + norm1_bias).squeeze(0)

        s, n, d = *norm1_out.shape[:1], self.num_heads, self.head_dim
        q = weights.self_attn_norm_q.apply(weights.self_attn_q.apply(norm1_out)).view(s, n, d)
        k = weights.self_attn_norm_k.apply(weights.self_attn_k.apply(norm1_out)).view(s, n, d)
        v = weights.self_attn_v.apply(norm1_out).view(s, n, d)


        freqs_i = compute_freqs_by_patch(q.size(0), q.size(2) // 2, grid_sizes, freqs, patch_index=self.patch_index, patch_num=self.patch_num)

        q = apply_rotary_emb(q, freqs_i)
        k = apply_rotary_emb(k, freqs_i)

        cu_seqlens_q, cu_seqlens_k = self._calculate_q_k_len(q, k_lens=seq_lens)
        
        if self.use_kv_cache:
            k, v = self.get_kv_cache_from_manager(cur_key=k, cur_value=v)

            attn_out = weights.self_attn_1.apply(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_k,
                max_seqlen_q=q.size(0),
                max_seqlen_kv=k.size(0),
                model_cls=self.config["model_cls"],
            )
        elif self.use_ring_attn_cache:
            # if self.rank == 0:
            #     import pdb; pdb.set_trace()
            # else:
            #     import time; time.sleep(99999)
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
            sub_patch_out, sub_patch_lse = ring_attn_sub(q, k, v)
            patch_out = self.ring_attn_cache_manager.get_full_out_with_cache(sub_patch_out, sub_patch_lse, self.patch_index, self.block_index)
            attn_out = patch_out

        y = weights.self_attn_o.apply(attn_out)
        x.add_(y * gate_msa.squeeze(0))
        return x

    def _infer_self_attn(self, weights, x, shift_msa, scale_msa, gate_msa, grid_sizes, freqs, seq_lens):
        full_x = x
        if self.use_kv_cache:
            if hasattr(weights, "smooth_norm1_weight"):
                norm1_weight = (1 + scale_msa) * weights.smooth_norm1_weight.tensor
                norm1_bias = shift_msa * weights.smooth_norm1_bias.tensor
            else:
                norm1_weight = 1 + scale_msa
                norm1_bias = shift_msa

            norm1_out = torch.nn.functional.layer_norm(x, (x.shape[1],), None, None, 1e-6)
            norm1_out = (norm1_out * norm1_weight + norm1_bias).squeeze(0)

            s, n, d = *norm1_out.shape[:1], self.num_heads, self.head_dim
            q = weights.self_attn_norm_q.apply(weights.self_attn_q.apply(norm1_out)).view(s, n, d)
            k = weights.self_attn_norm_k.apply(weights.self_attn_k.apply(norm1_out)).view(s, n, d)
            v = weights.self_attn_v.apply(norm1_out).view(s, n, d)

            freqs_i = compute_freqs(q.size(2) // 2, grid_sizes, freqs)

            q = apply_rotary_emb(q, freqs_i)
            k = apply_rotary_emb(k, freqs_i)

            cu_seqlens_q, cu_seqlens_k = self._calculate_q_k_len(q, k_lens=seq_lens)
            
            self.update_kv_cache_manager(full_key=k, full_value=v)
            
            # if self.rank == 0:
            #     import pdb; pdb.set_trace()
            # else:
            #     import time; time.sleep(9999)

            attn_out = weights.self_attn_1.apply(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_k,
                max_seqlen_q=q.size(0),
                max_seqlen_kv=k.size(0),
                model_cls=self.config["model_cls"],
            )
        elif self.use_ring_attn_cache:
            xs = self.get_patch_inputs(x)
            qs, ks, vs = [], [], []
            for patch_index, x in enumerate(xs):
                self.patch_index = patch_index
                if hasattr(weights, "smooth_norm1_weight"):
                    norm1_weight = (1 + scale_msa) * weights.smooth_norm1_weight.tensor
                    norm1_bias = shift_msa * weights.smooth_norm1_bias.tensor
                else:
                    norm1_weight = 1 + scale_msa
                    norm1_bias = shift_msa

                norm1_out = torch.nn.functional.layer_norm(x, (x.shape[1],), None, None, 1e-6)
                norm1_out = (norm1_out * norm1_weight + norm1_bias).squeeze(0)

                s, n, d = *norm1_out.shape[:1], self.num_heads, self.head_dim
                q = weights.self_attn_norm_q.apply(weights.self_attn_q.apply(norm1_out)).view(s, n, d)
                k = weights.self_attn_norm_k.apply(weights.self_attn_k.apply(norm1_out)).view(s, n, d)
                v = weights.self_attn_v.apply(norm1_out).view(s, n, d)

                freqs_i = compute_freqs_by_patch(q.size(0), q.size(2) // 2, grid_sizes, freqs, patch_index=self.patch_index, patch_num=self.patch_num)

                q = apply_rotary_emb(q, freqs_i)
                k = apply_rotary_emb(k, freqs_i)

                cu_seqlens_q, cu_seqlens_k = self._calculate_q_k_len(q, k_lens=seq_lens)
                
                q = q.unsqueeze(0)
                k = k.unsqueeze(0)
                v = v.unsqueeze(0)
                
                qs.append(q)
                ks.append(k)
                vs.append(v)
            
            full_out = []
            
            for q_patch_index, q in enumerate(qs):
                patch_out, patch_lse = None, None
                for kv_patch_index, (k, v) in enumerate(zip(ks, vs)):
                    sub_patch_out, sub_patch_lse = ring_attn_sub(q, k, v)
                
                    self.ring_attn_cache_manager.update_cache(sub_patch_out, sub_patch_lse, q_patch_index, kv_patch_index, self.block_index)
                    
                    patch_out, patch_lse = update_out_and_lse(patch_out, patch_lse, sub_patch_out, sub_patch_lse)
                full_out.append(patch_out)
                
            full_out = torch.cat(full_out, dim=1)
                    
            attn_out = full_out.to(torch.bfloat16).squeeze(0).reshape(q.shape[1]*self.patch_num, -1)
            
        y = weights.self_attn_o.apply(attn_out)
        full_x.add_(y * gate_msa.squeeze(0))
        return full_x