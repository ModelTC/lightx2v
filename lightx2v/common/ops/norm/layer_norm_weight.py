import torch
from abc import ABCMeta, abstractmethod
from lightx2v.utils.registry_factory import LN_WEIGHT_REGISTER


class LNWeightTemplate(metaclass=ABCMeta):
    def __init__(self, weight_name, bias_name, eps=1e-6):
        self.weight_name = weight_name
        self.bias_name = bias_name
        self.eps = eps
        self.config = {}

    def load(self, weight_dict):
        self.weight = weight_dict[self.weight_name].cuda() if self.weight_name is not None else None
        self.bias = weight_dict[self.bias_name].cuda() if self.bias_name is not None else None

    @abstractmethod
    def apply(self, input_tensor):
        pass

    def set_config(self, config=None):
        if config is not None:
            self.config = config

    def to_cpu(self, non_blocking=False):
        if self.weight is not None:
            self.weight = self.weight.to("cpu", non_blocking=non_blocking)
        if self.bias is not None:
            self.bias = self.bias.to("cpu", non_blocking=non_blocking)

    def to_cuda(self, non_blocking=False):
        if self.weight is not None:
            self.weight = self.weight.cuda(non_blocking=non_blocking)
        if self.bias is not None:
            self.bias = self.bias.cuda(non_blocking=non_blocking)

    def state_dict(self, destination=None):
        if destination is None:
            destination = {}
        if self.weight is not None:
            destination[self.weight_name] = self.weight.cpu().detach().clone()
        if self.bias is not None:
            destination[self.bias_name] = self.bias.cpu().detach().clone()
        return destination


@LN_WEIGHT_REGISTER("Default")
class LNWeight(LNWeightTemplate):
    def __init__(self, weight_name, bias_name, eps=1e-6):
        super().__init__(weight_name, bias_name, eps)

    def apply(self, input_tensor):
        input_tensor = torch.nn.functional.layer_norm(input_tensor, (input_tensor.shape[-1],), self.weight, self.bias, self.eps)
        return input_tensor
