import torch
import torch.distributed as dist
from lightx2v.utils.envs import *
from lightx2v.models.networks.wan.infer.transformer_infer import (
    WanTransformerInfer,
)
from lightx2v.attentions.distributed.comm.ring_comm import RingComm


class PipelineParallelWanTransformerInferWrapper:
    def __init__(self, transformer_infer: WanTransformerInfer, config):
        self.transformer_infer = transformer_infer
        self.config = config
        
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.ring_comm = RingComm()
        
    def set_blocks_num(self, blocks_num):
        self.transformer_infer.blocks_num = blocks_num

    def __getattr__(self, name: str):
        if name in self.__dict__:
            return getattr(self, name)
        else:
            return getattr(self.transformer_infer, name)
        
    # @torch.compile(disable=not CHECK_ENABLE_GRAPH_MODE())
    def infer(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        # for rank_step in range(self.world_size):
        #     if rank_step == self.rank:
        #         # import pdb; pdb.set_trace()
        #         print(f"cur rank: {self.rank}, cur step: {rank_step}")
        #         x = self.infer_func(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)
        #     # if rank_step != self.world_size - 1:
        #         x = self.ring_comm.send_recv(x)
        #         self.ring_comm.commit()
        #         self.ring_comm.wait()
        #     torch.cuda.synchronize()
        if self.rank == 0:
            x = self.infer_func(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)
            dist.send(x, dst=self.rank+1)
        elif self.rank == self.world_size-1:
            dist.recv(x, src=self.rank-1)
            x = self.infer_func(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)
        else:
            dist.recv(x, src=self.rank-1)
            x = self.infer_func(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)
            dist.send(x, dst=self.rank+1)
        
        dist.broadcast(x, src=self.world_size-1)
        return x