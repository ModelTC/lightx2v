import torch


class WeightAsyncStreamManager(object):
    def __init__(self):
        self.active_weights = [None for _ in range(2)]
        self.active_weights = [None for _ in range(2)]
        self.compute_stream = torch.cuda.Stream(priority=-1)
        self.load_stream = torch.cuda.Stream(priority=0)

    def prefetch_weights(self, block_idx, blocks_weights):
        with torch.cuda.stream(self.load_stream):
            if self.active_weights[1] is not None:
                self.active_weights[1].to_cpu_async()
            new_weights = blocks_weights[block_idx]
            new_weights.to_cuda_async()
            self.active_weights[1] = new_weights

    def swap_weights(self):
        self.compute_stream.synchronize()
        self.load_stream.synchronize()

        self.active_weights[0], self.active_weights[1] = (
            self.active_weights[1],
            self.active_weights[0],
        )

    def prefetch_phase(self, block_idx, phase_idx, blocks):
        with torch.cuda.stream(self.load_stream):
            if self.active_weights[1] is not None:
                _, old_phase = self.active_weights[1]
                old_phase.to_cpu_async()
            new_phase = blocks[block_idx].compute_phases[phase_idx]
            new_phase.to_cuda_async()
            self.active_weights[1] = (phase_idx, new_phase)

    def swap_phases(self):
        self.compute_stream.synchronize()
        self.load_stream.synchronize()
        self.active_weights[0], self.active_weights[1] = self.active_weights[1], self.active_weights[0]
