from ..transformer_infer import BaseHunyuanTransformerInfer
import torch

class HunyuanTransformerInferTeaCaching(BaseHunyuanTransformerInfer):
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
    

class HunyuanTransformerInferTaylorCaching(BaseHunyuanTransformerInfer):
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
            super().derivative_approximation(self.double_blocks_cache[i], "img_attn", img_out)
            super().derivative_approximation(self.double_blocks_cache[i], "txt_attn", txt_out)
            img, txt, img_out, txt_out, img_mod2_gate, txt_mod2_gate = super().infer_double_block_2(weights.double_blocks[i], img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec, frist_frame_token_num, img_out, txt_out, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate,tr_img_mod1_gate, tr_img_mod2_shift, tr_img_mod2_scale, tr_img_mod2_gate, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate)
            super().derivative_approximation(self.double_blocks_cache[i],"img_mlp", img_out)
            super().derivative_approximation(self.double_blocks_cache[i],"txt_mlp", txt_out)
            img, txt = super().infer_double_block_3(weights.double_blocks[i], img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec, frist_frame_token_num, img_out, txt_out, img_mod2_gate, txt_mod2_gate)

        x = torch.cat((img, txt), 0)
        for i in range(self.single_blocks_num):
            out, mod_gate, tr_mod_gate = super().infer_single_block_1(weights.single_blocks[i], x, vec, txt_seq_len, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec, frist_frame_token_num)
            super().derivative_approximation(self.single_blocks_cache[i], "total", out)
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
    
    # 1. taylor using caching
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


class HunyuanTransformerInferAdaCaching(BaseHunyuanTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        # 1. fixed args
        self.decisive_double_block_id = 10
        self.codebook = {0.03: 12, 0.05: 10, 0.07: 8, 0.09: 6, 0.11: 4, 1.00: 3}
        
        # 2. cache
        self.previous_residual_tiny = None
        self.now_residual_tiny = None
        self.norm_ord = 1
        self.skipped_step_length = 1
        self.previous_residual = None

        # 3. moreg
        self.previous_moreg = 1.
        self.moreg_strides = [1]
        self.moreg_steps = [
            int(0.1 * config.infer_steps),
            int(0.9 * config.infer_steps)
        ]
        self.moreg_hyp = [0.385, 8, 1, 2]
        self.mograd_mul = 10
        self.spatial_dim = 3072

    def infer(self, weights, img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec=None, frist_frame_token_num=None):
        # 1. 读取数组
        index = self.scheduler.step_index
        caching_records = self.scheduler.caching_records

        # 2. 判断完全计算，或者使用缓存
        if caching_records[index]:
            img, vec = self.infer_calculating(weights, img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis,token_replace_vec, frist_frame_token_num)

            # 3. 计算需要跳过的步数
            if index <= self.scheduler.infer_steps - 2:
                self.skipped_step_length = self.calculate_skip_step_length()
                for i in range(1, self.skipped_step_length):
                    if (index + i) <= self.scheduler.infer_steps - 1:
                        self.scheduler.caching_records[index+i] = False
        else:
            img, vec = self.infer_using_cache(weights, img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis,token_replace_vec, frist_frame_token_num)

        return img, vec

    def infer_calculating(self, weights, img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis,token_replace_vec, frist_frame_token_num):
        # 1. 拷贝噪声输入
        ori_img = img.clone()

        # 2. 完全计算
        txt_seq_len = txt.shape[0]
        img_seq_len = img.shape[0]
        for i in range(self.double_blocks_num):
            img_out, txt_out, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate,tr_img_mod1_gate, tr_img_mod2_shift, tr_img_mod2_scale, tr_img_mod2_gate, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate = super().infer_double_block_1(weights.double_blocks[i], img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec, frist_frame_token_num)
            img, txt, img_out, txt_out, img_mod2_gate, txt_mod2_gate = super().infer_double_block_2(weights.double_blocks[i], img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec, frist_frame_token_num, img_out, txt_out, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate,tr_img_mod1_gate, tr_img_mod2_shift, tr_img_mod2_scale, tr_img_mod2_gate, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate)
            if i == self.decisive_double_block_id:
                self.now_residual_tiny = img_out * img_mod2_gate
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

    # 1. ada's algorithm to calculate skip step length
    def calculate_skip_step_length(self):
        if self.previous_residual_tiny is None:
            self.previous_residual_tiny = self.now_residual_tiny
            return 1
        else:
            cache = self.previous_residual_tiny
            res = self.now_residual_tiny
            norm_ord = self.norm_ord
            cache_diff = (cache - res).norm(dim=(0,1), p=norm_ord) / cache.norm(dim=(0,1), p=norm_ord)
            cache_diff = cache_diff / self.skipped_step_length

            if self.moreg_steps[0] <= self.scheduler.step_index <= self.moreg_steps[1]:
                moreg = 0
                for i in self.moreg_strides:
                    moreg_i = (res[i*self.spatial_dim:, :] - res[:-i*self.spatial_dim, :]).norm(p=norm_ord) 
                    moreg_i /= (res[i*self.spatial_dim:, :].norm(p=norm_ord) + res[:-i*self.spatial_dim, :].norm(p=norm_ord))
                    moreg += moreg_i
                moreg = moreg / len(self.moreg_strides)
                moreg = ((1/self.moreg_hyp[0] * moreg) ** self.moreg_hyp[1]) / self.moreg_hyp[2] 
            else:
                moreg = 1.

            mograd = self.mograd_mul * (moreg - self.previous_moreg) / self.skipped_step_length
            self.previous_moreg = moreg
            moreg = moreg + abs(mograd)
            cache_diff = cache_diff * moreg

            metric_thres, cache_rates = list(self.codebook.keys()), list(self.codebook.values())
            if cache_diff < metric_thres[0]: new_rate = cache_rates[0]
            elif cache_diff < metric_thres[1]: new_rate = cache_rates[1]
            elif cache_diff < metric_thres[2]: new_rate = cache_rates[2]
            elif cache_diff < metric_thres[3]: new_rate = cache_rates[3]
            elif cache_diff < metric_thres[4]: new_rate = cache_rates[4]
            else: new_rate = cache_rates[-1]
            
            self.previous_residual_tiny = self.now_residual_tiny
            return new_rate


class HunyuanTransformerInferCustomCaching(BaseHunyuanTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self.teacache_thresh = self.config.teacache_thresh
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.previous_residual = None
        self.coefficients = [7.33226126e02, -4.01131952e02, 6.75869174e01, -3.14987800e00, 9.61237896e-02]

        self.cache = {}

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
        super().derivative_approximation(self.cache, "previous_residual", self.previous_residual)

        return img, vec

    def infer_using_cache(self, weights, img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec=None, frist_frame_token_num=None):
        img += super().taylor_formula(self.cache["previous_residual"])
        return img, vec