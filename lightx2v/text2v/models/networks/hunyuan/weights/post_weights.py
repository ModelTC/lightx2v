from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER
from lightx2v.common.ops.mm.mm_weight import MMWeightTemplate


class HunyuanPostWeights:
    def __init__(self, config):
        self.config = config

    def load_weights(self, weight_dict):
        self.final_layer_linear = MM_WEIGHT_REGISTER["Default-Force-FP32"]("final_layer.linear.weight", "final_layer.linear.bias")
        self.final_layer_adaLN_modulation_1 = MM_WEIGHT_REGISTER["Default"]("final_layer.adaLN_modulation.1.weight", "final_layer.adaLN_modulation.1.bias")

        self.weight_list = [
            self.final_layer_linear,
            self.final_layer_adaLN_modulation_1,
        ]

        for mm_weight in self.weight_list:
            if isinstance(mm_weight, MMWeightTemplate):
                mm_weight.set_config(self.config["mm_config"])
                mm_weight.load(weight_dict)

    def to_cpu(self):
        for mm_weight in self.weight_list:
            if isinstance(mm_weight, MMWeightTemplate):
                mm_weight.to_cpu()

    def to_cuda(self):
        for mm_weight in self.weight_list:
            if isinstance(mm_weight, MMWeightTemplate):
                mm_weight.to_cuda()
