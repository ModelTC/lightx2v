from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
import torch
from einops import rearrange
from .utils_bf16 import apply_rotary_emb
import numpy as np
from lightx2v.common.offload.manager import WeightAsyncStreamManager
from lightx2v.utils.envs import *

class BaseHunyuanTransformerInfer(BaseTransformerInfer):
    # 1. 初始化
    def __init__(self, config):
        self.config = config
        self.attention_type = config.get("attention_type", "flash_attn2")
        self.double_blocks_num = 20
        self.single_blocks_num = 40
        self.heads_num = 24
        self.hidden_size = 3072
        self.mlp_hidden_dim = 12288
        self.parallel_attention = None
        if self.config["cpu_offload"]:
            self.double_weights_stream_mgr = WeightAsyncStreamManager()
            self.single_weights_stream_mgr = WeightAsyncStreamManager()
            self.cpu_offload = True

    # per double block
    def infer_double_block_1(self, weights, img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec, frist_frame_token_num):
        # in this step, we finish both img and txt attention calculation before the first gate calculation
        # vec, scale, shift, gate
        vec_silu = torch.nn.functional.silu(vec)
        img_mod_out = weights.img_mod.apply(vec_silu)
        img_mod1_shift, img_mod1_scale, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate = img_mod_out.chunk(6, dim=-1)
        if token_replace_vec is not None:
            token_replace_vec_silu = torch.nn.functional.silu(token_replace_vec)
            token_replace_vec_img_mod_out = weights.img_mod.apply(token_replace_vec_silu)
            (tr_img_mod1_shift, tr_img_mod1_scale, tr_img_mod1_gate, tr_img_mod2_shift, tr_img_mod2_scale, tr_img_mod2_gate) = token_replace_vec_img_mod_out.chunk(6, dim=-1)
        else:
            (tr_img_mod1_shift, tr_img_mod1_scale, tr_img_mod1_gate, tr_img_mod2_shift, tr_img_mod2_scale, tr_img_mod2_gate) = None, None, None, None, None, None

        txt_mod_out = weights.txt_mod.apply(vec_silu)
        txt_mod1_shift, txt_mod1_scale, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate = txt_mod_out.chunk(6, dim=-1)

        # img: norm + scale&shift + qk-norm
        img_modulated = torch.nn.functional.layer_norm(img, (img.shape[1],), None, None, 1e-6)
        if tr_img_mod1_scale is not None:
            x_zero = img_modulated[:frist_frame_token_num] * (1 + tr_img_mod1_scale) + tr_img_mod1_shift
            x_orig = img_modulated[frist_frame_token_num:] * (1 + img_mod1_scale) + img_mod1_shift
            img_modulated = torch.concat((x_zero, x_orig), dim=0)
        else:
            img_modulated = img_modulated * (1 + img_mod1_scale) + img_mod1_shift
        img_qkv = weights.img_attn_qkv.apply(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "L (K H D) -> K L H D", K=3, H=self.heads_num)
        img_q = weights.img_attn_q_norm.apply(img_q)
        img_k = weights.img_attn_k_norm.apply(img_k)
        img_q, img_k = apply_rotary_emb(img_q, img_k, freqs_cis)

        # txt: norm + scale&shift + qk-norm
        txt_modulated = torch.nn.functional.layer_norm(txt, (txt.shape[1],), None, None, 1e-6)
        txt_modulated = txt_modulated * (1 + txt_mod1_scale) + txt_mod1_shift
        txt_qkv = weights.txt_attn_qkv.apply(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "L (K H D) -> K L H D", K=3, H=self.heads_num)
        txt_q = weights.txt_attn_q_norm.apply(txt_q)
        txt_k = weights.txt_attn_k_norm.apply(txt_k)

        # attention
        q = torch.cat((img_q, txt_q), dim=0)
        k = torch.cat((img_k, txt_k), dim=0)
        v = torch.cat((img_v, txt_v), dim=0)
        if not self.parallel_attention:
            attn = weights.double_attn.apply(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_qkv,
                cu_seqlens_kv=cu_seqlens_qkv,
                max_seqlen_q=max_seqlen_qkv,
                max_seqlen_kv=max_seqlen_qkv,
            )
        else:
            # world_size = dist.get_world_size()
            attn = self.parallel_attention(
                attention_type=self.attention_type,
                q=q,
                k=k,
                v=v,
                img_qkv_len=img_q.shape[0],
                cu_seqlens_qkv=cu_seqlens_qkv,
                # cu_seqlens_qkv=cu_seqlens_qkv,
                # max_seqlen_qkv=max_seqlen_qkv,
            )
        img_attn, txt_attn = attn[: img.shape[0]], attn[img.shape[0] :]
        img_out = weights.img_attn_proj.apply(img_attn)
        txt_out = weights.txt_attn_proj.apply(txt_attn)
        return img_out, txt_out, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate,tr_img_mod1_gate, tr_img_mod2_shift, tr_img_mod2_scale, tr_img_mod2_gate, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate
        
    def infer_double_block_2(self, weights, img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec, frist_frame_token_num, img_out, txt_out, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate,tr_img_mod1_gate, tr_img_mod2_shift, tr_img_mod2_scale, tr_img_mod2_gate, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate):
        # img: gate + norm + scale&shift + mlp
        if tr_img_mod1_gate is not None:
            x_zero =  img_out[:frist_frame_token_num] * tr_img_mod1_gate
            x_orig =  img_out[frist_frame_token_num:] * img_mod1_gate
            img_out = torch.concat((x_zero, x_orig), dim=0)
        else:
            img_out =  img_out * img_mod1_gate
        img = img + img_out

        img_out = torch.nn.functional.layer_norm(img, (img.shape[1],), None, None, 1e-6)

        if tr_img_mod1_gate is not None:
            x_zero = img_out[:frist_frame_token_num] * (1 + tr_img_mod2_scale) + tr_img_mod2_shift
            x_orig = img_out[frist_frame_token_num:] * (1 + img_mod2_scale) + img_mod2_shift
            img_out = torch.concat((x_zero, x_orig), dim=0)
        else:
            img_out = img_out * (1 + img_mod2_scale) + img_mod2_shift

        img_out = weights.img_mlp_fc1.apply(img_out)
        img_out = torch.nn.functional.gelu(img_out, approximate="tanh")
        img_out = weights.img_mlp_fc2.apply(img_out)

        # txt: gate + norm + scale&shift + mlp
        txt_out = txt_out * txt_mod1_gate
        txt = txt + txt_out

        txt_out = torch.nn.functional.layer_norm(txt, (txt.shape[1],), None, None, 1e-6)

        txt_out = txt_out * (1 + txt_mod2_scale) + txt_mod2_shift

        txt_out = weights.txt_mlp_fc1.apply(txt_out)
        txt_out = torch.nn.functional.gelu(txt_out, approximate="tanh")
        txt_out = weights.txt_mlp_fc2.apply(txt_out)

        return img, txt, img_out, txt_out, img_mod2_gate, txt_mod2_gate
        
    def infer_double_block_3(self, weights, img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec, frist_frame_token_num, img_out, txt_out, img_mod2_gate, txt_mod2_gate):
        # img
        img_out = img_out * img_mod2_gate
        img = img + img_out

        # txt
        txt_out = txt_out * txt_mod2_gate
        txt = txt + txt_out

        return img, txt

    # per single blcok
    def infer_single_block_1(self, weights, x, vec, txt_seq_len, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec=None, frist_frame_token_num=None):
        # vec, scale, shift, gate
        out = torch.nn.functional.silu(vec)
        out = weights.modulation.apply(out)
        mod_shift, mod_scale, mod_gate = out.chunk(3, dim=-1)

        if token_replace_vec is not None:
            token_replace_vec_out = torch.nn.functional.silu(token_replace_vec)
            token_replace_vec_out = weights.modulation.apply(token_replace_vec_out)
            tr_mod_shift, tr_mod_scale, tr_mod_gate = token_replace_vec_out.chunk(3, dim=-1)
        else:
            tr_mod_shift, tr_mod_scale, tr_mod_gate = None, None, None

        # norm + scale&shift + qk-norm + attention
        out = torch.nn.functional.layer_norm(x, (x.shape[1],), None, None, 1e-6)

        if token_replace_vec is not None:
            x_zero = out[:frist_frame_token_num] * (1 + tr_mod_scale) + tr_mod_shift
            x_orig = out[frist_frame_token_num:] * (1 + mod_scale) + mod_shift
            x_mod = torch.concat((x_zero, x_orig), dim=0)
        else:
            x_mod = out * (1 + mod_scale) + mod_shift

        x_mod = weights.linear1.apply(x_mod)

        qkv, mlp = torch.split(x_mod, [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
        q, k, v = rearrange(qkv, "L (K H D) -> K L H D", K=3, H=self.heads_num)
        q = weights.q_norm.apply(q)
        k = weights.k_norm.apply(k)

        img_q, txt_q = q[:-txt_seq_len, :, :], q[-txt_seq_len:, :, :]
        img_k, txt_k = k[:-txt_seq_len, :, :], k[-txt_seq_len:, :, :]
        img_q, img_k = apply_rotary_emb(img_q, img_k, freqs_cis)
        q = torch.cat((img_q, txt_q), dim=0)
        k = torch.cat((img_k, txt_k), dim=0)
        if not self.parallel_attention:
            attn = weights.single_attn.apply(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_qkv,
                cu_seqlens_kv=cu_seqlens_qkv,
                max_seqlen_q=max_seqlen_qkv,
                max_seqlen_kv=max_seqlen_qkv,
            )
        else:
            attn = self.parallel_attention(
                attention_type=self.attention_type,
                q=q,
                k=k,
                v=v,
                img_qkv_len=img_q.shape[0],
                cu_seqlens_qkv=cu_seqlens_qkv,
                # cu_seqlens_qkv=cu_seqlens_qkv,
                # max_seqlen_qkv=max_seqlen_qkv,
            )

        # mlp
        out = torch.nn.functional.gelu(mlp, approximate="tanh")

        # cat
        out = torch.cat((attn, out), 1)
        out = weights.linear2.apply(out)

        return out, mod_gate, tr_mod_gate

    def infer_single_block_2(self, weights, x, vec, txt_seq_len, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, out, mod_gate, tr_mod_gate, token_replace_vec=None, frist_frame_token_num=None):
        if token_replace_vec is not None:
            x_zero = out[:frist_frame_token_num] * tr_mod_gate
            x_orig = out[frist_frame_token_num:] * mod_gate
            out = torch.concat((x_zero, x_orig), dim=0)
        else:
            out = out * mod_gate
        x = x + out
        return x

    # 1. only in tea-cache, judge next step
    def calculate_should_calc(self, img, vec, weights):
        # 1. 时间步嵌入调制
        inp = img.clone()
        vec_ = vec.clone()
        img_mod1_shift, img_mod1_scale, _, _, _, _ = weights.double_blocks[0].img_mod.apply(vec_).chunk(6, dim=-1)
        normed_inp = torch.nn.functional.layer_norm(inp, (inp.shape[1],), None, None, 1e-6)
        modulated_inp = normed_inp * (1 + img_mod1_scale) + img_mod1_shift
        del normed_inp, inp, vec_

        # 2. L1距离计算
        if self.scheduler.step_index == 0 or self.scheduler.step_index == self.scheduler.infer_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            rescale_func = np.poly1d(self.coefficients)
            self.accumulated_rel_l1_distance += rescale_func(
                ((modulated_inp -  self.previous_modulated_input).abs().mean() /  self.previous_modulated_input.abs().mean()).cpu().item()
            )
            if self.accumulated_rel_l1_distance < self.teacache_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        del modulated_inp

        # 3. 返回判断
        return should_calc

    # 1. get taylor step_diff when there is only on caching_records in scheduler
    def get_taylor_step_diff(self):
        current_step = self.scheduler.step_index
        last_calc_step = current_step - 1
        while last_calc_step >= 0 and not self.scheduler.caching_records[last_calc_step]:
            last_calc_step -= 1
        step_diff = current_step - last_calc_step
        return step_diff


class HunyuanTransformerInfer(BaseHunyuanTransformerInfer):
    def __init__(self, config):
        super().__init__(config)

    @torch.compile(disable=not CHECK_ENABLE_GRAPH_MODE())
    def infer(self, weights, img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec=None, frist_frame_token_num=None):
        # 1. 读取数组
        index = self.scheduler.step_index
        caching_records = self.scheduler.caching_records

        # 2. 判断完全计算，或者使用缓存
        if caching_records[index]:
            return self.infer_calculating(weights, img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis,token_replace_vec, frist_frame_token_num)
        else:
            return self.infer_using_cache(weights, img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis,token_replace_vec, frist_frame_token_num)

    def infer_calculating(self, weights, img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec=None, frist_frame_token_num=None):
        txt_seq_len = txt.shape[0]
        img_seq_len = img.shape[0]
        for i in range(self.double_blocks_num):
            img_out, txt_out, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate,tr_img_mod1_gate, tr_img_mod2_shift, tr_img_mod2_scale, tr_img_mod2_gate, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate = super().infer_double_block_1(weights.double_blocks[i], img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec, frist_frame_token_num)
            img, txt, img_out, txt_out, img_mod2_gate, txt_mod2_gate = super().infer_double_block_2(weights.double_blocks[i], img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec, frist_frame_token_num, img_out, txt_out, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate,tr_img_mod1_gate, tr_img_mod2_shift, tr_img_mod2_scale, tr_img_mod2_gate, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate)
            img, txt = super().infer_double_block_3(weights.double_blocks[i], img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec, frist_frame_token_num, img_out, txt_out, img_mod2_gate, txt_mod2_gate)

        x = torch.cat((img, txt), 0)
        for i in range(self.single_blocks_num):
            out, mod_gate, tr_mod_gate = super().infer_single_block_1(weights.single_blocks[i], x, vec, txt_seq_len, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec, frist_frame_token_num)
            x = super().infer_single_block_2(weights.single_blocks[i], x, vec, txt_seq_len, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, out, mod_gate, tr_mod_gate, token_replace_vec, frist_frame_token_num)
        img = x[:img_seq_len, ...]
        return img, vec
        
    def infer_using_cache(self, weights, img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec=None, frist_frame_token_num=None):
        pass