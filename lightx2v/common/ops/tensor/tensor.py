import torch
from lightx2v.utils.registry_factory import TENSOR_REGISTER


@TENSOR_REGISTER("Default")
class DefaultTensor:
    def __init__(self, tensor_name):
        self.tensor_name = tensor_name

    def load(self, weight_dict):
        self.tensor = weight_dict[self.tensor_name]
        self.pinned_tensor = torch.empty(self.tensor.shape, pin_memory=True, dtype=self.tensor.dtype)

    def to_cpu(self, non_blocking=False):
        # self.tensor = self.tensor.to("cpu", non_blocking=non_blocking)
        self.tensor = self.pinned_tensor.copy_(self.tensor, non_blocking=non_blocking).cpu()

    def to_cuda(self, non_blocking=False):
        self.tensor = self.tensor.cuda(non_blocking=non_blocking)

    def state_dict(self, destination=None):
        if destination is None:
            destination = {}
        destination[self.tensor_name] = self.tensor.cpu().detach().clone()
        return destination
