import torch
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER, LN_WEIGHT_REGISTER, CONV3D_WEIGHT_REGISTER
from lightx2v.common.ops.mm.mm_weight import MMWeightTemplate
from lightx2v.common.ops.norm.layer_norm_weight import LNWeightTemplate
from lightx2v.common.ops.conv.conv3d import Conv3dWeightTemplate


class WanPreWeights:
    def __init__(self, config):
        self.in_dim = config["in_dim"]
        self.dim = config["dim"]
        self.patch_size = (1, 2, 2)
        self.config = config

    def load_weights(self, weight_dict):
        self.patch_embedding = CONV3D_WEIGHT_REGISTER["Defaultt-Force-BF16"]("patch_embedding.weight", "patch_embedding.bias", stride=self.patch_size)

        self.text_embedding_0 = MM_WEIGHT_REGISTER["Default"]("condition_embedder.text_embedder.linear_1.weight", "condition_embedder.text_embedder.linear_1.bias")
        self.text_embedding_2 = MM_WEIGHT_REGISTER["Default"]("condition_embedder.text_embedder.linear_2.weight", "condition_embedder.text_embedder.linear_2.bias")
        
        self.time_embedding_0 = MM_WEIGHT_REGISTER["Default"]("condition_embedder.time_embedder.linear_1.weight", "condition_embedder.time_embedder.linear_1.bias")
        self.time_embedding_2 = MM_WEIGHT_REGISTER["Default"]("condition_embedder.time_embedder.linear_2.weight", "condition_embedder.time_embedder.linear_2.bias")
        
        
        self.time_projection_1 = MM_WEIGHT_REGISTER["Default"]("condition_embedder.time_proj.weight", "condition_embedder.time_proj.bias")

        self.weight_list = [
            self.patch_embedding,
            self.text_embedding_0,
            self.text_embedding_2,
            self.time_embedding_0,
            self.time_embedding_2,
            self.time_projection_1,
        ]

        if self.config['task'] == 'i2v':
            self.proj_0 = LN_WEIGHT_REGISTER["Default"]("condition_embedder.image_embedder.norm1.weight", "condition_embedder.image_embedder.norm1.bias", eps=1e-5)
            
            
            self.proj_1 = MM_WEIGHT_REGISTER["Default"]("condition_embedder.image_embedder.ff.net.0.proj.weight", "condition_embedder.image_embedder.ff.net.0.proj.bias")
            self.proj_3 = MM_WEIGHT_REGISTER["Default"]("condition_embedder.image_embedder.ff.net.2.weight", "condition_embedder.image_embedder.ff.net.2.bias")
            
            self.proj_4 = LN_WEIGHT_REGISTER["Default"]("condition_embedder.image_embedder.norm2.weight", "condition_embedder.image_embedder.norm2.bias", eps=1e-5)
            
            self.weight_list.append(self.proj_0)
            self.weight_list.append(self.proj_1)
            self.weight_list.append(self.proj_3)
            self.weight_list.append(self.proj_4)

        for mm_weight in self.weight_list:
            if isinstance(mm_weight, (MMWeightTemplate, LNWeightTemplate, Conv3dWeightTemplate)):
                mm_weight.set_config(self.config["mm_config"])
                mm_weight.load(weight_dict)
                if self.config["cpu_offload"]:
                    mm_weight.to_cpu()

    def to_cpu(self):
        for mm_weight in self.weight_list:
            if isinstance(mm_weight, (MMWeightTemplate, LNWeightTemplate, Conv3dWeightTemplate)):
                mm_weight.to_cpu()

    def to_cuda(self):
        for mm_weight in self.weight_list:
            if isinstance(mm_weight, (MMWeightTemplate, LNWeightTemplate, Conv3dWeightTemplate)):
                mm_weight.to_cuda()
