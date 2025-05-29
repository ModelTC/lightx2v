from ..transformer_infer import BaseHunyuanTransformer
import torch

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
            should_calc = super().calculate_should_calc(img, vec, weights)
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
