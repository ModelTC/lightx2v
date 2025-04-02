import torch
from abc import ABCMeta, abstractmethod
from vllm import _custom_ops as ops
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER
from lightx2v.utils.quant_utils import IntegerQuantizer, FloatQuantizer

try:
    import q8_kernels.functional as Q8F
except ImportError:
    Q8F = None


class MMWeightTemplate(metaclass=ABCMeta):
    def __init__(self, weight_name, bias_name):
        self.weight_name = weight_name
        self.bias_name = bias_name
        self.config = {}

    @abstractmethod
    def load(self, weight_dict):
        pass

    @abstractmethod
    def apply(self, input_tensor):
        pass

    def set_config(self, config=None):
        if config is not None:
            self.config = config


@MM_WEIGHT_REGISTER("Default")
class MMWeight(MMWeightTemplate):
    def __init__(self, weight_name, bias_name):
        super().__init__(weight_name, bias_name)

    def load(self, weight_dict):
        self.weight = weight_dict[self.weight_name].t().cuda()
        self.bias = weight_dict[self.bias_name].cuda() if self.bias_name is not None else None

    def apply(self, input_tensor):
        shape = (input_tensor.shape[0], self.weight.shape[1])
        dtype = input_tensor.dtype
        device = input_tensor.device
        output_tensor = torch.empty(shape, dtype=dtype, device=device, requires_grad=False)
        if self.bias is None:
            return torch.mm(input_tensor, self.weight, out=output_tensor)
        return torch.addmm(self.bias, input_tensor, self.weight, out=output_tensor)

    def to_cpu(self):
        self.weight = self.weight.cpu()
        if self.bias is not None:
            self.bias = self.bias.cpu()

    def to_cuda(self):
        self.weight = self.weight.cuda()
        if self.bias is not None:
            self.bias = self.bias.cuda()


@MM_WEIGHT_REGISTER("Default-Force-FP32")
class MMWeightForceFP32(MMWeight):
    def __init__(self, weight_name, bias_name):
        super().__init__(weight_name, bias_name)

    def load(self, weight_dict):
        super().load(weight_dict)
        self.weight = self.weight.to(torch.float32)
        if self.bias is not None:
            self.bias = self.bias.to(torch.float32)


@MM_WEIGHT_REGISTER("W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Vllm")
class MMWeightWfp8channelAfp8channeldynamicVllm(MMWeightTemplate):
    """
    Name: W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Vllm

    Quant MM:
        Weight: fp8 perchannel sym
        Act: fp8 perchannel dynamic sym
        Kernel: vllm
    """

    def __init__(self, weight_name, bias_name):
        super().__init__(weight_name, bias_name)

    def load(self, weight_dict):
        if self.config.get("weight_auto_quant", True):
            self.weight = weight_dict[self.weight_name].to(torch.float32).cuda()
            w_quantizer = FloatQuantizer("e4m3", True, "channel")
            self.weight, self.weight_scale, _ = w_quantizer.real_quant_tensor(self.weight)
            self.weight = self.weight.to(torch.float8_e4m3fn).t().cuda()
            self.weight_scale = self.weight_scale.to(torch.float32).cuda()
        else:
            self.weight = weight_dict[self.weight_name].t().cuda()
            self.weight_scale = weight_dict[self.weight_name.rstrip(".weight") + ".weight_scale"].cuda()
        self.bias = weight_dict[self.bias_name].cuda() if self.bias_name is not None else None

    def apply(self, input_tensor):
        shape = (input_tensor.shape[0], self.weight.shape[1])
        dtype = input_tensor.dtype
        device = input_tensor.device
        output_tensor = torch.empty(shape, dtype=dtype, device=device, requires_grad=False)
        qinput, x_scale = ops.scaled_fp8_quant(input_tensor, None, scale_ub=None, use_per_token_if_dynamic=True)
        torch.ops._C.cutlass_scaled_mm(output_tensor, qinput, self.weight, x_scale, self.weight_scale, self.bias)
        return output_tensor

    def to_cpu(self):
        self.weight = self.weight.cpu()
        self.weight_scale = self.weight_scale.cpu()
        if self.bias is not None:
            self.bias = self.bias.cpu()

    def to_cuda(self):
        self.weight = self.weight.cuda()
        self.weight_scale = self.weight_scale.cuda()
        if self.bias is not None:
            self.bias = self.bias.cuda()


@MM_WEIGHT_REGISTER("W-int8-channel-sym-A-int8-channel-sym-dynamic-Vllm")
class MMWeightWint8channelAint8channeldynamicVllm(MMWeightTemplate):
    """
    Name: W-int8-channel-sym-A-int8-channel-sym-dynamic-Vllm

    Quant MM:
        Weight: int8 perchannel sym
        Act: int8 perchannel dynamic sym
        Kernel: vllm
    """

    def __init__(self, weight_name, bias_name):
        super().__init__(weight_name, bias_name)

    def load(self, weight_dict):
        if self.config.get("weight_auto_quant", True):
            self.weight = weight_dict[self.weight_name].to(torch.float32).cuda()
            w_quantizer = IntegerQuantizer(8, True, "channel")
            self.weight, self.weight_scale, _ = w_quantizer.real_quant_tensor(self.weight)
            self.weight = self.weight.to(torch.int8).t().cuda()
            self.weight_scale = self.weight_scale.to(torch.float32).cuda()
        else:
            self.weight = weight_dict[self.weight_name].t().cuda()
            self.weight_scale = weight_dict[self.weight_name.rstrip(".weight") + ".weight_scale"].cuda()
        self.bias = weight_dict[self.bias_name].cuda() if self.bias_name is not None else None

    def apply(self, input_tensor):
        shape = (input_tensor.shape[0], self.weight.shape[1])
        dtype = input_tensor.dtype
        device = input_tensor.device
        output_tensor = torch.empty(shape, dtype=dtype, device=device, requires_grad=False)
        qinput, x_scale, _ = ops.scaled_int8_quant(input_tensor, scale=None, azp=None, symmetric=True)
        torch.ops._C.cutlass_scaled_mm(output_tensor, qinput, self.weight, x_scale, self.weight_scale, self.bias)
        return output_tensor

    def to_cpu(self):
        self.weight = self.weight.cpu()
        self.weight_scale = self.weight_scale.cpu()
        if self.bias is not None:
            self.bias = self.bias.cpu()

    def to_cuda(self):
        self.weight = self.weight.cuda()
        self.weight_scale = self.weight_scale.cuda()
        if self.bias is not None:
            self.bias = self.bias.cuda()


@MM_WEIGHT_REGISTER("W-int8-channel-sym-A-int8-channel-sym-dynamic-Q8F")
class MMWeightWint8channelAint8channeldynamicQ8F(MMWeightTemplate):
    """
    Name: W-int8-channel-sym-A-int8-channel-sym-dynamic-Q8F

    Quant MM:
        Weight: int8 perchannel sym
        Act: int8 perchannel dynamic sym
        Kernel: Q8F
    """

    def __init__(self, weight_name, bias_name):
        super().__init__(weight_name, bias_name)

    def load(self, weight_dict):
        if self.config.get("weight_auto_quant", True):
            self.weight = weight_dict[self.weight_name].cuda()
            w_quantizer = IntegerQuantizer(8, True, "channel")
            self.weight, self.weight_scale, _ = w_quantizer.real_quant_tensor(self.weight)
            self.weight = self.weight.to(torch.int8)
            self.weight_scale = self.weight_scale.to(torch.float32)
        else:
            self.weight = weight_dict[self.weight_name].cuda()
            self.weight_scale = weight_dict[self.weight_name.rstrip(".weight") + ".weight_scale"].cuda()
        self.bias = weight_dict[self.bias_name].float().cuda() if self.bias_name is not None else None

    def apply(self, input_tensor, act=None):
        qinput, x_scale, _ = ops.scaled_int8_quant(input_tensor, scale=None, azp=None, symmetric=True)
        output_tensor = Q8F.linear.q8_linear(qinput, self.weight, self.bias, x_scale, self.weight_scale, fuse_gelu=False, out_dtype=torch.bfloat16)
        return output_tensor.squeeze(0)

    def to_cpu(self):
        self.weight = self.weight.cpu()
        self.weight_scale = self.weight_scale.cpu()
        if self.bias is not None:
            self.bias = self.bias.cpu()

    def to_cuda(self):
        self.weight = self.weight.cuda()
        self.weight_scale = self.weight_scale.cuda()
        if self.bias is not None:
            self.bias = self.bias.cuda()


@MM_WEIGHT_REGISTER("W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Q8F")
class MMWeightWfp8channelAfp8channeldynamicQ8F(MMWeightTemplate):
    """
    Name: W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Q8F

    Quant MM:
        Weight: fp8 perchannel sym
        Act: fp8 perchannel dynamic sym
        Kernel: Q8F
    """

    def __init__(self, weight_name, bias_name):
        super().__init__(weight_name, bias_name)

    def load(self, weight_dict):
        if self.config.get("weight_auto_quant", True):
            self.weight = weight_dict[self.weight_name].cuda()
            w_quantizer = FloatQuantizer("e4m3", True, "channel")
            self.weight, self.weight_scale, _ = w_quantizer.real_quant_tensor(self.weight)
            self.weight = self.weight.to(torch.float8_e4m3fn)
            self.weight_scale = self.weight_scale.to(torch.float32)
        else:
            self.weight = weight_dict[self.weight_name].cuda()
            self.weight_scale = weight_dict[self.weight_name.rstrip(".weight") + ".weight_scale"].cuda()
        self.bias = weight_dict[self.bias_name].float().cuda() if self.bias_name is not None else None

    def apply(self, input_tensor):
        qinput, x_scale = ops.scaled_fp8_quant(input_tensor, None, scale_ub=None, use_per_token_if_dynamic=True)
        output_tensor = Q8F.linear.fp8_linear(qinput, self.weight, self.bias, x_scale, self.weight_scale, out_dtype=torch.bfloat16)
        return output_tensor.squeeze(0)

    def to_cpu(self):
        self.weight = self.weight.cpu()
        self.weight_scale = self.weight_scale.cpu()
        if self.bias is not None:
            self.bias = self.bias.cpu()

    def to_cuda(self):
        self.weight = self.weight.cuda()
        self.weight_scale = self.weight_scale.cuda()
        if self.bias is not None:
            self.bias = self.bias.cuda()


if __name__ == "__main__":
    weight_dict = {
        "xx.weight": torch.randn(8192, 4096).to(torch.float8_e4m3fn),
        "xx.bias": torch.randn(8192).to(torch.bfloat16),
        "xx.weight_scale": torch.randn(8192, 1).to(torch.float32),
    }

    mm_weight = MM_WEIGHT_REGISTER["W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Vllm"]("xx.weight", "xx.bias")
    mm_weight.set_config({"weight_auto_quant": False})
    mm_weight.load(weight_dict)
    input_tensor = torch.randn(1024, 4096).to(torch.bfloat16).cuda()
    output_tensor = mm_weight.apply(input_tensor)
    print(output_tensor.shape)

    weight_dict = {
        "xx.weight": torch.randn(8192, 4096),
        "xx.bias": torch.randn(8192).to(torch.bfloat16),
    }

    mm_weight = MM_WEIGHT_REGISTER["W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Vllm"]("xx.weight", "xx.bias")
    mm_weight.set_config({"weight_auto_quant": True})
    mm_weight.load(weight_dict)
    input_tensor = torch.randn(1024, 4096).to(torch.bfloat16).cuda()
    output_tensor = mm_weight.apply(input_tensor)
    print(output_tensor.shape)

    weight_dict = {
        "xx.weight": torch.randn(8192, 4096),
        "xx.bias": torch.randn(8192).to(torch.bfloat16),
    }

    mm_weight = MM_WEIGHT_REGISTER["W-int8-channel-sym-A-int8-channel-sym-dynamic-Vllm"]("xx.weight", "xx.bias")
    mm_weight.set_config({"weight_auto_quant": True})
    mm_weight.load(weight_dict)
    input_tensor = torch.randn(1024, 4096).to(torch.bfloat16).cuda()
    output_tensor = mm_weight.apply(input_tensor)
    print(output_tensor.shape)
