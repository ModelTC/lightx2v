import torch
from einops import rearrange
from lightx2v.attentions import attention
from .utils_bf16 import apply_rotary_emb


class HunyuanTransformerInfer:
    def __init__(self, config):
        self.config = config
        self.attention_type = config.get("attention_type", "flash_attn2")
        self.double_blocks_num = 20
        self.single_blocks_num = 40
        self.heads_num = 24
        self.hidden_size = 3072
        self.mlp_hidden_dim = 12288
        self.parallel_attention = None

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def infer(self, weights, img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis):
        txt_seq_len = txt.shape[0]
        img_seq_len = img.shape[0]

        for i in range(self.double_blocks_num):
            img, txt = self.infer_double_block(
                weights.double_blocks_weights[i],
                img,
                txt,
                vec,
                cu_seqlens_qkv,
                max_seqlen_qkv,
                freqs_cis,
            )

        x = torch.cat((img, txt), 0)

        for i in range(self.single_blocks_num):
            x = self.infer_single_block(
                weights.single_blocks_weights[i],
                x,
                vec,
                txt_seq_len,
                cu_seqlens_qkv,
                max_seqlen_qkv,
                freqs_cis,
            )

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

        img_q, img_k, img_v = self.infer_double_block_img_pre_atten(weights, img, img_mod1_scale, img_mod1_shift, freqs_cis)
        txt_q, txt_k, txt_v = self.infer_double_block_txt_pre_atten(weights, txt, txt_mod1_scale, txt_mod1_shift)

        q = torch.cat((img_q, txt_q), dim=0)
        k = torch.cat((img_k, txt_k), dim=0)
        v = torch.cat((img_v, txt_v), dim=0)

        if not self.parallel_attention:
            attn = attention(
                attention_type=self.attention_type,
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
        img = self.infer_double_block_img_post_atten(weights, img, img_attn, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate)
        txt = self.infer_double_block_txt_post_atten(weights, txt, txt_attn, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate)
        return img, txt

    def infer_double_block_img_pre_atten(self, weights, img, img_mod1_scale, img_mod1_shift, freqs_cis):
        img_modulated = torch.nn.functional.layer_norm(img, (img.shape[1],), None, None, 1e-6)
        img_modulated = img_modulated * (1 + img_mod1_scale) + img_mod1_shift
        img_qkv = weights.img_attn_qkv.apply(img_modulated)

        img_q, img_k, img_v = rearrange(img_qkv, "L (K H D) -> K L H D", K=3, H=self.heads_num)

        img_q = weights.img_attn_q_norm.apply(img_q)
        img_k = weights.img_attn_k_norm.apply(img_k)

        img_q, img_k = apply_rotary_emb(img_q, img_k, freqs_cis)
        return img_q, img_k, img_v

    def infer_double_block_txt_pre_atten(self, weights, txt, txt_mod1_scale, txt_mod1_shift):
        txt_modulated = torch.nn.functional.layer_norm(txt, (txt.shape[1],), None, None, 1e-6)
        txt_modulated = txt_modulated * (1 + txt_mod1_scale) + txt_mod1_shift
        txt_qkv = weights.txt_attn_qkv.apply(txt_modulated)

        txt_q, txt_k, txt_v = rearrange(txt_qkv, "L (K H D) -> K L H D", K=3, H=self.heads_num)

        txt_q = weights.txt_attn_q_norm.apply(txt_q)
        txt_k = weights.txt_attn_k_norm.apply(txt_k)
        return txt_q, txt_k, txt_v

    def infer_double_block_img_post_atten(self, weights, img, img_attn, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate):
        out = weights.img_attn_proj.apply(img_attn)
        out = out * img_mod1_gate
        img = img + out

        out = torch.nn.functional.layer_norm(img, (img.shape[1],), None, None, 1e-6)
        out = out * (1 + img_mod2_scale) + img_mod2_shift
        out = weights.img_mlp_fc1.apply(out)
        out = torch.nn.functional.gelu(out, approximate="tanh")
        out = weights.img_mlp_fc2.apply(out)
        out = out * img_mod2_gate
        img = img + out
        return img

    def infer_double_block_txt_post_atten(self, weights, txt, txt_attn, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate):
        out = weights.txt_attn_proj.apply(txt_attn)
        out = out * txt_mod1_gate
        txt = txt + out

        out = torch.nn.functional.layer_norm(txt, (txt.shape[1],), None, None, 1e-6)
        out = out * (1 + txt_mod2_scale) + txt_mod2_shift
        out = weights.txt_mlp_fc1.apply(out)
        out = torch.nn.functional.gelu(out, approximate="tanh")
        out = weights.txt_mlp_fc2.apply(out)
        out = out * txt_mod2_gate
        txt = txt + out
        return txt

    def infer_single_block(self, weights, x, vec, txt_seq_len, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis):
        out = torch.nn.functional.silu(vec)
        out = weights.modulation.apply(out)
        mod_shift, mod_scale, mod_gate = out.chunk(3, dim=-1)

        out = torch.nn.functional.layer_norm(x, (x.shape[1],), None, None, 1e-6)
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
            attn = attention(
                attention_type=self.attention_type,
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

        out = torch.nn.functional.gelu(mlp, approximate="tanh")
        out = torch.cat((attn, out), 1)
        out = weights.linear2.apply(out)
        out = out * mod_gate
        x = x + out
        return x
