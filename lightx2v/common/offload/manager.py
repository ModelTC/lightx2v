import torch


class WeightStreamManager(object):
    def __init__(self, blocks_num, offload_ratio=1):
        self.active_weights = [None for _ in range(3)]
        self.compute_stream = torch.cuda.Stream(priority=-1)
        self.cpu_load_stream = torch.cuda.Stream(priority=0)
        self.cuda_load_stream = torch.cuda.Stream(priority=0)
        self.offload_block_num = offload_ratio * blocks_num

    def prefetch_weights(self, block_idx, blocks_weights):
        with torch.cuda.stream(self.cuda_load_stream):
            self.active_weights[2] = blocks_weights[block_idx]
            self.active_weights[2].to_cuda_sync()
        with torch.cuda.stream(self.cpu_load_stream):
            if block_idx < self.offload_block_num:
                if self.active_weights[1] is not None:
                    self.active_weights[1].to_cpu_sync()

    def swap_weights(self):
        self.compute_stream.synchronize()
        self.cpu_load_stream.synchronize()
        self.cuda_load_stream.synchronize()

        self.active_weights[0], self.active_weights[1] = (
            self.active_weights[2],
            self.active_weights[0],
        )
