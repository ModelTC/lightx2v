import torch
import torch.nn as nn
# from vllm import _custom_ops as ops

try:
    import q8_kernels.functional as Q8F
except ImportError:
    Q8F = None


class QuantLinearInt8(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer("weight", torch.empty((out_features, in_features), dtype=torch.int8))
        self.register_buffer("weight_scale", torch.empty((out_features, 1), dtype=torch.float32))

        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=torch.float32))
        else:
            self.register_buffer("bias", None)

    def act_quant_func(self, x):
        # input_tensor_quant, input_tensor_scale, _ = ops.scaled_int8_quant(x, scale=None, azp=None, symmetric=True)
        # return input_tensor_quant, input_tensor_scale
        pass

    def forward(self, x):
        input_tensor_quant, input_tensor_scale = self.act_quant_func(x)
        output_tensor = Q8F.linear.q8_linear(
            input_tensor_quant,
            self.weight,
            self.bias.float() if self.bias is not None else None,
            input_tensor_scale,
            self.weight_scale.float(),
            fuse_gelu=False,
            out_dtype=torch.bfloat16,
        )
        return output_tensor
