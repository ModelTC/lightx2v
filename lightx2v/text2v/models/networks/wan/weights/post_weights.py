from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER
from lightx2v.common.ops.mm.mm_weight import MMWeightTemplate


class WanPostWeights:
    def __init__(self, config):
        self.config = config

    def load_weights(self, weight_dict):
        self.head = MM_WEIGHT_REGISTER["Default"]("head.head.weight", "head.head.bias")
        self.head_modulation = weight_dict["head.modulation"]

        self.weight_list = [self.head]

        for mm_weight in self.weight_list:
            if isinstance(mm_weight, MMWeightTemplate):
                mm_weight.set_config(self.config["mm_config"])
                mm_weight.load(weight_dict)
                if self.config["cpu_offload"]:
                    mm_weight.to_cpu()
                    self.head_modulation = self.head_modulation.cpu()

    def to_cpu(self):
        for mm_weight in self.weight_list:
            if isinstance(mm_weight, MMWeightTemplate):
                mm_weight.to_cpu()
        self.head_modulation = self.head_modulation.cpu()

    def to_cuda(self):
        for mm_weight in self.weight_list:
            if isinstance(mm_weight, MMWeightTemplate):
                mm_weight.to_cuda()
        self.head_modulation = self.head_modulation.cuda()
