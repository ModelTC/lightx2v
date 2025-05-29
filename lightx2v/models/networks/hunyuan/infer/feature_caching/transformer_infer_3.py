from ..transformer_infer import BaseHunyuanTransformer
import torch
import numpy as np

class HunyuanTransformerInferTaylorCaching(BaseHunyuanTransformer):
    def __init__(self, config):
        super().__init__(config)
        self.double_blocks_cache = [{} for _ in range(self.double_blocks_num)]
        self.single_blocks_cache = [{} for _ in range(self.single_blocks_num)]

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
            self.derivative_approximation(self.double_blocks_cache[i], "img_attn", img_out)
            self.derivative_approximation(self.double_blocks_cache[i], "txt_attn", txt_out)
            img, txt, img_out, txt_out, img_mod2_gate, txt_mod2_gate = super().infer_double_block_2(weights.double_blocks[i], img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec, frist_frame_token_num, img_out, txt_out, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate,tr_img_mod1_gate, tr_img_mod2_shift, tr_img_mod2_scale, tr_img_mod2_gate, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate)
            self.derivative_approximation(self.double_blocks_cache[i],"img_mlp", img_out)
            self.derivative_approximation(self.double_blocks_cache[i],"txt_mlp", txt_out)
            img, txt = super().infer_double_block_3(weights.double_blocks[i], img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec, frist_frame_token_num, img_out, txt_out, img_mod2_gate, txt_mod2_gate)

        x = torch.cat((img, txt), 0)
        for i in range(self.single_blocks_num):
            out, mod_gate, tr_mod_gate = super().infer_single_block_1(weights.single_blocks[i], x, vec, txt_seq_len, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec, frist_frame_token_num)
            self.derivative_approximation(self.single_blocks_cache[i], "total", out)
            x = super().infer_single_block_2(weights.single_blocks[i], x, vec, txt_seq_len, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, out, mod_gate, tr_mod_gate, token_replace_vec, frist_frame_token_num)
        img = x[:img_seq_len, ...]
        return img, vec
        
    def infer_using_cache(self, weights, img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec=None, frist_frame_token_num=None):
        txt_seq_len = txt.shape[0]
        img_seq_len = img.shape[0]
        for i in range(self.double_blocks_num):
            img, txt = self.infer_double_block(weights.double_blocks[i], img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, i)

        x = torch.cat((img, txt), 0)
        for i in range(self.single_blocks_num):
            x = self.infer_single_block(weights.single_blocks[i], x, vec, txt_seq_len, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, i)

        img = x[:img_seq_len, ...]
        return img, vec
    
    # 1. when fully calcualted, stored in cache
    def derivative_approximation(self, block_cache, module_name, out):
        if module_name not in block_cache:
            block_cache[module_name] = {0: out}
        else:
            step_diff = super().get_taylor_step_diff()

            previous_out = block_cache[module_name][0]
            block_cache[module_name][0] = out
            block_cache[module_name][1] = (out - previous_out) / step_diff

    # 2. taylor using caching
    def infer_double_block(self, weights, img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, i):
        vec_silu = torch.nn.functional.silu(vec)
        img_mod_out = weights.img_mod.apply(vec_silu)
        img_mod1_shift, img_mod1_scale, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate = img_mod_out.chunk(6, dim=-1)
        txt_mod_out = weights.txt_mod.apply(vec_silu)
        txt_mod1_shift, txt_mod1_scale, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate = txt_mod_out.chunk(6, dim=-1)

        out = super().taylor_formula(self.double_blocks_cache[i]["img_attn"])
        out = out * img_mod1_gate
        img = img + out

        out = super().taylor_formula(self.double_blocks_cache[i]["img_mlp"])
        out = out * img_mod2_gate
        img = img + out

        out = super().taylor_formula(self.double_blocks_cache[i]["txt_attn"])
        out = out * txt_mod1_gate
        txt = txt + out

        out = super().taylor_formula(self.double_blocks_cache[i]["txt_mlp"])
        out = out * txt_mod2_gate
        txt = txt + out

        return img, txt

    def infer_single_block(self, weights, x, vec, txt_seq_len, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, i):
        out = torch.nn.functional.silu(vec)
        out = weights.modulation.apply(out)
        mod_shift, mod_scale, mod_gate = out.chunk(3, dim=-1)

        out = super().taylor_formula(self.single_blocks_cache[i]["total"])
        out = out * mod_gate
        x = x + out
        return x
    
