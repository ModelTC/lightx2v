import torch.distributed as dist
from lightx2v.models.networks.wan.weights.transformer_weights import (
    WanTransformerWeights,
)


class PipelineParallelWanTransformerWeightsWrapper:
    def __init__(self, transformer_weights: WanTransformerWeights, config):
        self.transformer_weights = transformer_weights
        self.config = config
        
        self._split_weights_by_rank(self.transformer_weights)
        
    def get_blocks_num(self):
        return self.blocks_num
        
    def _split_weights_by_rank(self, transformer_weights):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        transformer_weights_blocks_num = len(transformer_weights.blocks)
        
        base_blocks_num = transformer_weights_blocks_num // world_size
        remainder_blocks_num = transformer_weights_blocks_num % world_size
        
        start = 0
        block_idx_list = [0]
        
        for i in range(world_size):
            end = start + base_blocks_num + (1 if i <remainder_blocks_num else 0)
            block_idx_list.append(end)
            start = end
        
        transformer_weights.blocks = transformer_weights.blocks[block_idx_list[rank]:block_idx_list[rank+1]]
        self.blocks_num = len(transformer_weights.blocks)

    def __getattr__(self, name: str):
        if name in self.__dict__:
            return getattr(self, name)
        else:
            return getattr(self.transformer_weights, name)
        
    