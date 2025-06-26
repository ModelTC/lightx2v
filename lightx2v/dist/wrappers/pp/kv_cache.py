from lightx2v.attentions.distributed.ring.attn import update_out_and_lse
import torch


class PipelineParallelKVCacheManager:
    def __init__(self, kv_cache_len, patch_num):
        self.kv_cache_len = kv_cache_len
        self.patch_num = patch_num
        self.key_cache_list = [[None for i in range(self.patch_num)] for j in range(self.kv_cache_len)]
        self.value_cache_list = [[None for i in range(self.patch_num)] for j in range(self.kv_cache_len)]
        self.cur_index = 0
        
    # def _update_index(self):
    #     self.cur_index = (self.cur_index+1) % self.kv_cache_len
        
    def update_kv_cache(self, key, value, patch_index, block_index):
        self.key_cache_list[block_index][patch_index] = key
        self.value_cache_list[block_index][patch_index] = value
        # self._update_index()
        
    def get_full_kv_with_cache(self, key, value, patch_index, block_index):
        assert patch_index < self.patch_num, "patch_index should be smaller than patch_num"
        # self.key_cache_list[block_index][patch_index] = key
        # self.value_cache_list[block_index][patch_index] = value
        self.update_kv_cache(key, value, patch_index, block_index)
        # TODO certainfy cat dim!!!!
        full_key = torch.cat(self.key_cache_list[block_index], dim=0)
        full_value = torch.cat(self.value_cache_list[block_index], dim=0)
        return full_key, full_value


class RingAttnCacheManager:
    def __init__(self, cache_len, patch_num):
        self.cache_len = cache_len
        self.patch_num = patch_num
        self.patch_out_list = [[[None for _ in range(patch_num)] for _ in range(patch_num)] for _ in range(cache_len)]
        self.patch_lse_list = [[[None for _ in range(patch_num)] for _ in range(patch_num)] for _ in range(cache_len)]
        
    def update_cache(self, patch_out, patch_lse, q_patch_index, kv_patch_index,  block_index):
        self.patch_out_list[block_index][q_patch_index][kv_patch_index] = patch_out
        self.patch_lse_list[block_index][q_patch_index][kv_patch_index] = patch_lse
    
    def get_full_out_with_cache(self, sub_patch_out, sub_patch_lse, patch_index, block_index):
        assert patch_index < self.patch_num, "patch_index should be smaller than patch_num"
        self.update_cache(sub_patch_out, sub_patch_lse, patch_index, patch_index, block_index)
        patch_out = None
        patch_lse = None
        for i in range(self.patch_num):
            out = self.patch_out_list[block_index][patch_index][i]
            lse = self.patch_lse_list[block_index][patch_index][i]
            patch_out, patch_lse = update_out_and_lse(patch_out, patch_lse, out, lse)
        return patch_out.to(torch.bfloat16).squeeze(0).reshape(sub_patch_out.shape[1], -1)