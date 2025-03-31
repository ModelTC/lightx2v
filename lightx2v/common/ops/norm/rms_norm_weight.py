import torch
from abc import ABCMeta, abstractmethod
from lightx2v.utils.registry_factory import RMS_WEIGHT_REGISTER
import sgl_kernel


class RMSWeightTemplate(metaclass=ABCMeta):
    def __init__(self, weight_name, eps=1e-6):
        self.weight_name = weight_name
        self.eps = eps
        self.config = {}

    def load(self, weight_dict):
        self.weight = weight_dict[self.weight_name].cuda()

    @abstractmethod
    def apply(self, input_tensor):
        pass

    def set_config(self, config=None):
        if config is not None:
            self.config = config

    def to_cpu(self):
        self.weight = self.weight.cpu()

    def to_cuda(self):
        self.weight = self.weight.cuda()


@RMS_WEIGHT_REGISTER("Default")
class RMSWeight(RMSWeightTemplate):
    def __init__(self, weight_name, eps=1e-6):
        super().__init__(weight_name, eps)

    def apply(self, input_tensor):
        input_tensor = input_tensor * torch.rsqrt(
            input_tensor.pow(2).mean(-1, keepdim=True) + self.eps
        )
        input_tensor = input_tensor * self.weight
        return input_tensor


@RMS_WEIGHT_REGISTER("FP32")
class RMSWeightFP32(RMSWeight):
    def __init__(self, weight_name, eps=1e-6):
        super().__init__(weight_name, eps)

    def apply(self, input_tensor):
        input_tensor = input_tensor.float()
        input_tensor = input_tensor * torch.rsqrt(
            input_tensor.pow(2).mean(-1, keepdim=True) + self.eps
        )
        input_tensor = input_tensor.to(torch.bfloat16)
        input_tensor = input_tensor * self.weight
        return input_tensor


@RMS_WEIGHT_REGISTER("sgl-kernel")
class RMSWeightSgl(RMSWeight):
    def __init__(self, weight_name, eps=1e-6):
        super().__init__(weight_name, eps)

    def apply(self, input_tensor):
        input_tensor = input_tensor.contiguous()
        orig_shape = input_tensor.shape
        input_tensor = input_tensor.view(-1, orig_shape[-1])
        input_tensor = sgl_kernel.rmsnorm(input_tensor, self.weight, self.eps).view(orig_shape)
        return input_tensor
