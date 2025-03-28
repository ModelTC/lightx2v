from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER
from lightx2v.common.ops.mm.mm_weight import MMWeightTemplate


class WanPostWeights:
    def __init__(self, config):
        self.config = config

    def load_weights(self, weight_dict):
        # 1. 加载weight
        self.head = MM_WEIGHT_REGISTER["Default"]('head.head.weight','head.head.bias')
        self.head_modulation = weight_dict['head.modulation']

        self.weight_list = [
            self.head,
            self.head_modulation
        ]

        for mm_weight in self.weight_list:
            if isinstance(mm_weight, MMWeightTemplate):
                mm_weight.set_config(self.config['mm_config'])
                mm_weight.load(weight_dict)

    def to_cpu(self):
        for mm_weight in self.weight_list:
            if isinstance(mm_weight, MMWeightTemplate):
                mm_weight.to_cpu()
            else:
                mm_weight.cpu()

    def to_cuda(self):
        for mm_weight in self.weight_list:
            if isinstance(mm_weight, MMWeightTemplate):
                mm_weight.to_cuda()
            else:
                mm_weight.cuda()