import torch


class WeightStreamManager(object):
    def __init__(self):
        self.active_weights = [None for _ in range(2)]
        self.compute_stream = torch.cuda.Stream(priority=-1)
        self.load_stream = torch.cuda.Stream(priority=0)

    def prefetch_weights(self, block_idx, blocks_weights):
        with torch.cuda.stream(self.load_stream):
            if self.active_weights[1] is not None:
                self.active_weights[1].to_cpu_sync()
            new_weights = blocks_weights[block_idx]
            new_weights.to_cuda_sync()
            self.active_weights[1] = new_weights

    def swap_weights(self):
        self.compute_stream.synchronize()
        self.load_stream.synchronize()

        self.active_weights[0], self.active_weights[1] = (
            self.active_weights[1],
            self.active_weights[0],
        )
