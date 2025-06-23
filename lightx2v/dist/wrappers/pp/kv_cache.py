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
        