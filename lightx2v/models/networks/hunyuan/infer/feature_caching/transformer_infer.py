import torch
import numpy as np
from einops import rearrange
from .utils import taylor_cache_init, derivative_approximation, taylor_formula
from ..utils_bf16 import apply_rotary_emb
from ..transformer_infer import HunyuanTransformerInfer


class HunyuanTransformerInferTeaCaching(HunyuanTransformerInfer):
    def __init__(self, config):
        super().__init__(config)

    def infer(
        self,
        weights,
        img,
        txt,
        vec,
        cu_seqlens_qkv,
        max_seqlen_qkv,
        freqs_cis,
        token_replace_vec=None,
        frist_frame_token_num=None,
    ):
        inp = img.clone()
        vec_ = vec.clone()

        weights.double_blocks[0].to_cuda()
        img_mod1_shift, img_mod1_scale, _, _, _, _ = weights.double_blocks[0].img_mod.apply(vec_).chunk(6, dim=-1)
        weights.double_blocks[0].to_cpu_sync()

        normed_inp = torch.nn.functional.layer_norm(inp, (inp.shape[1],), None, None, 1e-6)
        modulated_inp = normed_inp * (1 + img_mod1_scale) + img_mod1_shift
        del normed_inp, inp, vec_

        if self.scheduler.step_index == 0 or self.scheduler.step_index == self.scheduler.infer_steps - 1:
            should_calc = True
            self.scheduler.accumulated_rel_l1_distance = 0
        else:
            rescale_func = np.poly1d(self.scheduler.coefficients)
            self.scheduler.accumulated_rel_l1_distance += rescale_func(
                ((modulated_inp - self.scheduler.previous_modulated_input).abs().mean() / self.scheduler.previous_modulated_input.abs().mean()).cpu().item()
            )
            if self.scheduler.accumulated_rel_l1_distance < self.scheduler.teacache_thresh:
                should_calc = False
            else:
                should_calc = True
                self.scheduler.accumulated_rel_l1_distance = 0
        self.scheduler.previous_modulated_input = modulated_inp
        del modulated_inp

        if not should_calc:
            img += self.scheduler.previous_residual
        else:
            ori_img = img.clone()
            img, vec = super().infer(weights, img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec, frist_frame_token_num)
            self.scheduler.previous_residual = img - ori_img
            del ori_img
            torch.cuda.empty_cache()

        return img, vec


class HunyuanTransformerInferTaylorCaching(HunyuanTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        assert not self.config["cpu_offload"], "Not support cpu-offload for TaylorCaching"

    def infer(self, weights, img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec=None, frist_frame_token_num=None):
        txt_seq_len = txt.shape[0]
        img_seq_len = img.shape[0]

        self.scheduler.current["stream"] = "double_stream"
        for i in range(self.double_blocks_num):
            self.scheduler.current["layer"] = i
            img, txt = self.infer_double_block(weights.double_blocks[i], img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis)

        x = torch.cat((img, txt), 0)

        self.scheduler.current["stream"] = "single_stream"
        for i in range(self.single_blocks_num):
            self.scheduler.current["layer"] = i
            x = self.infer_single_block(weights.single_blocks[i], x, vec, txt_seq_len, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis)

        img = x[:img_seq_len, ...]
        return img, vec

    def infer_double_block(self, weights, img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis):
        vec_silu = torch.nn.functional.silu(vec)

        img_mod_out = weights.img_mod.apply(vec_silu)
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = img_mod_out.chunk(6, dim=-1)

        txt_mod_out = weights.txt_mod.apply(vec_silu)
        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = txt_mod_out.chunk(6, dim=-1)

        if self.scheduler.current["type"] == "full":
            img_q, img_k, img_v = self.infer_double_block_img_pre_atten(weights, img, img_mod1_scale, img_mod1_shift, None, None, None, freqs_cis)
            txt_q, txt_k, txt_v = self.infer_double_block_txt_pre_atten(weights, txt, txt_mod1_scale, txt_mod1_shift)

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
            img = self.infer_double_block_img_post_atten(
                weights,
                img,
                img_attn,
                img_mod1_gate,
                img_mod2_shift,
                img_mod2_scale,
                img_mod2_gate,
            )
            txt = self.infer_double_block_txt_post_atten(
                weights,
                txt,
                txt_attn,
                txt_mod1_gate,
                txt_mod2_shift,
                txt_mod2_scale,
                txt_mod2_gate,
            )
            return img, txt

        elif self.scheduler.current["type"] == "taylor_cache":
            self.scheduler.current["module"] = "img_attn"

            out = taylor_formula(self.scheduler.cache_dic, self.scheduler.current)

            out = out * img_mod1_gate
            img = img + out

            self.scheduler.current["module"] = "img_mlp"

            out = taylor_formula(self.scheduler.cache_dic, self.scheduler.current)

            out = out * img_mod2_gate
            img = img + out

            self.scheduler.current["module"] = "txt_attn"

            out = taylor_formula(self.scheduler.cache_dic, self.scheduler.current)

            out = out * txt_mod1_gate
            txt = txt + out

            self.scheduler.current["module"] = "txt_mlp"

            out = out * txt_mod2_gate
            txt = txt + out

            return img, txt

    def infer_double_block_img_post_atten(
        self,
        weights,
        img,
        img_attn,
        img_mod1_gate,
        img_mod2_shift,
        img_mod2_scale,
        img_mod2_gate,
    ):
        self.scheduler.current["module"] = "img_attn"
        taylor_cache_init(self.scheduler.cache_dic, self.scheduler.current)

        out = weights.img_attn_proj.apply(img_attn)
        derivative_approximation(self.scheduler.cache_dic, self.scheduler.current, out)

        out = out * img_mod1_gate
        img = img + out

        self.scheduler.current["module"] = "img_mlp"
        taylor_cache_init(self.scheduler.cache_dic, self.scheduler.current)

        out = torch.nn.functional.layer_norm(img, (img.shape[1],), None, None, 1e-6)
        out = out * (1 + img_mod2_scale) + img_mod2_shift
        out = weights.img_mlp_fc1.apply(out)
        out = torch.nn.functional.gelu(out, approximate="tanh")
        out = weights.img_mlp_fc2.apply(out)
        derivative_approximation(self.scheduler.cache_dic, self.scheduler.current, out)

        out = out * img_mod2_gate
        img = img + out
        return img

    def infer_double_block_txt_post_atten(
        self,
        weights,
        txt,
        txt_attn,
        txt_mod1_gate,
        txt_mod2_shift,
        txt_mod2_scale,
        txt_mod2_gate,
    ):
        self.scheduler.current["module"] = "txt_attn"
        taylor_cache_init(self.scheduler.cache_dic, self.scheduler.current)

        out = weights.txt_attn_proj.apply(txt_attn)
        derivative_approximation(self.scheduler.cache_dic, self.scheduler.current, out)

        out = out * txt_mod1_gate
        txt = txt + out

        self.scheduler.current["module"] = "txt_mlp"
        taylor_cache_init(self.scheduler.cache_dic, self.scheduler.current)

        out = torch.nn.functional.layer_norm(txt, (txt.shape[1],), None, None, 1e-6)
        out = out * (1 + txt_mod2_scale) + txt_mod2_shift
        out = weights.txt_mlp_fc1.apply(out)
        out = torch.nn.functional.gelu(out, approximate="tanh")
        out = weights.txt_mlp_fc2.apply(out)
        derivative_approximation(self.scheduler.cache_dic, self.scheduler.current, out)

        out = out * txt_mod2_gate
        txt = txt + out
        return txt

    def infer_single_block(self, weights, x, vec, txt_seq_len, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis):
        out = torch.nn.functional.silu(vec)
        out = weights.modulation.apply(out)
        mod_shift, mod_scale, mod_gate = out.chunk(3, dim=-1)

        if self.scheduler.current["type"] == "full":
            out = torch.nn.functional.layer_norm(x, (x.shape[1],), None, None, 1e-6)
            x_mod = out * (1 + mod_scale) + mod_shift
            x_mod = weights.linear1.apply(x_mod)
            qkv, mlp = torch.split(x_mod, [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

            self.scheduler.current["module"] = "attn"
            taylor_cache_init(self.scheduler.cache_dic, self.scheduler.current)

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

            derivative_approximation(self.scheduler.cache_dic, self.scheduler.current, attn)
            self.scheduler.current["module"] = "total"
            taylor_cache_init(self.scheduler.cache_dic, self.scheduler.current)

            out = torch.nn.functional.gelu(mlp, approximate="tanh")
            out = torch.cat((attn, out), 1)
            out = weights.linear2.apply(out)
            derivative_approximation(self.scheduler.cache_dic, self.scheduler.current, out)

            out = out * mod_gate
            x = x + out
            return x

        elif self.scheduler.current["type"] == "taylor_cache":
            self.scheduler.current["module"] = "total"
            out = taylor_formula(self.scheduler.cache_dic, self.scheduler.current)
            out = out * mod_gate
            x = x + out
            return x
