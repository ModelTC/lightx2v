from ..transformer_infer import BaseHunyuanTransformer
import torch
import numpy as np

class HunyuanTransformerInferTeaCaching(BaseHunyuanTransformer):
    def __init__(self, config):
        super().__init__(config)
        self.teacache_thresh = self.config.teacache_thresh
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.previous_residual = None
        self.coefficients = [7.33226126e02, -4.01131952e02, 6.75869174e01, -3.14987800e00, 9.61237896e-02]

    def infer(self, weights, img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec=None, frist_frame_token_num=None):
        # 1. 读取数组
        index = self.scheduler.step_index
        caching_records = self.scheduler.caching_records

        # 2. 判断完全计算，或者使用缓存
        if caching_records[index]:
            img, vec = self.infer_calculating(weights, img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis,token_replace_vec, frist_frame_token_num)
        else:
            img, vec = self.infer_using_cache(weights, img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis,token_replace_vec, frist_frame_token_num)
        
        # 3. 更新数组
        if index <= self.scheduler.infer_steps - 2:
            should_calc = self.calculate_should_calc(img, vec, weights)
            self.scheduler.caching_records[index+1] = should_calc

        return img, vec

    def infer_calculating(self, weights, img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec=None, frist_frame_token_num=None):
        # 1. 拷贝噪声输入
        ori_img = img.clone()

        # 2. 完全计算
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

        # 3. 缓存残差到Transformer实例中
        self.previous_residual=img - ori_img

        return img, vec

    def infer_using_cache(self, weights, img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec=None, frist_frame_token_num=None):
        img += self.previous_residual
        return img, vec

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
