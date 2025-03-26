from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER, LN_WEIGHT_REGISTER, CONV3D_WEIGHT_REGISTER
from lightx2v.common.ops.mm.mm_weight import MMWeightTemplate
from lightx2v.common.ops.norm.layer_norm_weight import LNWeightTemplate
from lightx2v.common.ops.conv.conv3d import Conv3dWeightTemplate


class WanPostWeights:
    def __init__(self):
        pass

    def load_weights(self, weight_dict):
        # 1. 加载weight
        self.head = MM_WEIGHT_REGISTER["Default"]('head.head.weight','head.head.bias')
        self.head_modulation = weight_dict['head.modulation']

        self.weight_list = [
            self.head,
            self.head_modulation
        ]

        for mm_weight in self.weight_list:
            if isinstance(mm_weight, MMWeightTemplate) or isinstance(mm_weight, LNWeightTemplate) or isinstance(mm_weight, Conv3dWeightTemplate):
                mm_weight.set_config(self.config['mm_config'])
                mm_weight.load(weight_dict)

    def to_cpu(self):
        for mm_weight in self.weight_list:
            if isinstance(mm_weight, MMWeightTemplate) or isinstance(mm_weight, LNWeightTemplate) or isinstance(mm_weight, Conv3dWeightTemplate):
                mm_weight.to_cpu()
            else:
                mm_weight.cpu()

    def to_cuda(self):
        for mm_weight in self.weight_list:
            if isinstance(mm_weight, MMWeightTemplate) or isinstance(mm_weight, LNWeightTemplate) or isinstance(mm_weight, Conv3dWeightTemplate):
                mm_weight.to_cuda()
            else:
                mm_weight.cuda()