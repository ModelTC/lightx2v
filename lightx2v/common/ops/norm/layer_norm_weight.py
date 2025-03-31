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

    def to_cpu(self):
        if self.weight is not None:
            self.weight = self.weight.cpu()
        if self.bias is not None:
            self.bias = self.bias.cpu()

    def to_cuda(self):
        if self.weight is not None:
            self.weight = self.weight.cuda()
        if self.bias is not None:
            self.bias = self.bias.cuda()


@LN_WEIGHT_REGISTER("Default")
class LNWeight(LNWeightTemplate):
    def __init__(self, weight_name, bias_name, eps=1e-6):
        super().__init__(weight_name, bias_name, eps)

    def apply(self, input_tensor):
        input_tensor = torch.nn.functional.layer_norm(
            input_tensor, (input_tensor.shape[1],), self.weight, self.bias, self.eps
        )
        return input_tensor
