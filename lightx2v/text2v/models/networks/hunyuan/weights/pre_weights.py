from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER, LN_WEIGHT_REGISTER, CONV3D_WEIGHT_REGISTER
from lightx2v.common.ops.mm.mm_weight import MMWeightTemplate
from lightx2v.common.ops.norm.layer_norm_weight import LNWeightTemplate
from lightx2v.common.ops.conv.conv3d import Conv3dWeightTemplate


class HunyuanPreWeights:
    def __init__(self, config):
        self.config = config

    def load_weights(self, weight_dict):
        self.img_in_proj = CONV3D_WEIGHT_REGISTER["Default"]('img_in.proj.weight', 'img_in.proj.bias', stride=(1, 2, 2))

        self.txt_in_input_embedder = MM_WEIGHT_REGISTER["Default"]('txt_in.input_embedder.weight', 'txt_in.input_embedder.bias')
        self.txt_in_t_embedder_mlp_0 = MM_WEIGHT_REGISTER["Default"]('txt_in.t_embedder.mlp.0.weight', 'txt_in.t_embedder.mlp.0.bias')
        self.txt_in_t_embedder_mlp_2 = MM_WEIGHT_REGISTER["Default"]('txt_in.t_embedder.mlp.2.weight', 'txt_in.t_embedder.mlp.2.bias')
        self.txt_in_c_embedder_linear_1 = MM_WEIGHT_REGISTER["Default"]('txt_in.c_embedder.linear_1.weight', 'txt_in.c_embedder.linear_1.bias')
        self.txt_in_c_embedder_linear_2 = MM_WEIGHT_REGISTER["Default"]('txt_in.c_embedder.linear_2.weight', 'txt_in.c_embedder.linear_2.bias')

        self.txt_in_individual_token_refiner_blocks_0_norm1 = LN_WEIGHT_REGISTER["Default"]('txt_in.individual_token_refiner.blocks.0.norm1.weight', 'txt_in.individual_token_refiner.blocks.0.norm1.bias', eps=1e-6)
        self.txt_in_individual_token_refiner_blocks_0_self_attn_qkv = MM_WEIGHT_REGISTER["Default"]('txt_in.individual_token_refiner.blocks.0.self_attn_qkv.weight', 'txt_in.individual_token_refiner.blocks.0.self_attn_qkv.bias')
        self.txt_in_individual_token_refiner_blocks_0_self_attn_proj = MM_WEIGHT_REGISTER["Default"]('txt_in.individual_token_refiner.blocks.0.self_attn_proj.weight', 'txt_in.individual_token_refiner.blocks.0.self_attn_proj.bias')
        self.txt_in_individual_token_refiner_blocks_0_norm2 = LN_WEIGHT_REGISTER["Default"]('txt_in.individual_token_refiner.blocks.0.norm2.weight', 'txt_in.individual_token_refiner.blocks.0.norm2.bias', eps=1e-6)
        self.txt_in_individual_token_refiner_blocks_0_mlp_fc1 = MM_WEIGHT_REGISTER["Default"]('txt_in.individual_token_refiner.blocks.0.mlp.fc1.weight', 'txt_in.individual_token_refiner.blocks.0.mlp.fc1.bias')
        self.txt_in_individual_token_refiner_blocks_0_mlp_fc2 = MM_WEIGHT_REGISTER["Default"]('txt_in.individual_token_refiner.blocks.0.mlp.fc2.weight', 'txt_in.individual_token_refiner.blocks.0.mlp.fc2.bias')
        self.txt_in_individual_token_refiner_blocks_0_adaLN_modulation_1 = MM_WEIGHT_REGISTER["Default"]('txt_in.individual_token_refiner.blocks.0.adaLN_modulation.1.weight', 'txt_in.individual_token_refiner.blocks.0.adaLN_modulation.1.bias')

        self.txt_in_individual_token_refiner_blocks_1_norm1 = LN_WEIGHT_REGISTER["Default"]('txt_in.individual_token_refiner.blocks.1.norm1.weight', 'txt_in.individual_token_refiner.blocks.1.norm1.bias', eps=1e-6)
        self.txt_in_individual_token_refiner_blocks_1_self_attn_qkv = MM_WEIGHT_REGISTER["Default"]('txt_in.individual_token_refiner.blocks.1.self_attn_qkv.weight', 'txt_in.individual_token_refiner.blocks.1.self_attn_qkv.bias')
        self.txt_in_individual_token_refiner_blocks_1_self_attn_proj = MM_WEIGHT_REGISTER["Default"]('txt_in.individual_token_refiner.blocks.1.self_attn_proj.weight', 'txt_in.individual_token_refiner.blocks.1.self_attn_proj.bias')
        self.txt_in_individual_token_refiner_blocks_1_norm2 = LN_WEIGHT_REGISTER["Default"]('txt_in.individual_token_refiner.blocks.1.norm2.weight', 'txt_in.individual_token_refiner.blocks.1.norm2.bias', eps=1e-6)
        self.txt_in_individual_token_refiner_blocks_1_mlp_fc1 = MM_WEIGHT_REGISTER["Default"]('txt_in.individual_token_refiner.blocks.1.mlp.fc1.weight', 'txt_in.individual_token_refiner.blocks.1.mlp.fc1.bias')
        self.txt_in_individual_token_refiner_blocks_1_mlp_fc2 = MM_WEIGHT_REGISTER["Default"]('txt_in.individual_token_refiner.blocks.1.mlp.fc2.weight', 'txt_in.individual_token_refiner.blocks.1.mlp.fc2.bias')
        self.txt_in_individual_token_refiner_blocks_1_adaLN_modulation_1 = MM_WEIGHT_REGISTER["Default"]('txt_in.individual_token_refiner.blocks.1.adaLN_modulation.1.weight', 'txt_in.individual_token_refiner.blocks.1.adaLN_modulation.1.bias')
        
        self.time_in_mlp_0 = MM_WEIGHT_REGISTER["Default"]('time_in.mlp.0.weight', 'time_in.mlp.0.bias')
        self.time_in_mlp_2 = MM_WEIGHT_REGISTER["Default"]('time_in.mlp.2.weight', 'time_in.mlp.2.bias')
        self.vector_in_in_layer = MM_WEIGHT_REGISTER["Default"]('vector_in.in_layer.weight', 'vector_in.in_layer.bias')
        self.vector_in_out_layer = MM_WEIGHT_REGISTER["Default"]('vector_in.out_layer.weight', 'vector_in.out_layer.bias')
        self.guidance_in_mlp_0 = MM_WEIGHT_REGISTER["Default"]('guidance_in.mlp.0.weight', 'guidance_in.mlp.0.bias')
        self.guidance_in_mlp_2 = MM_WEIGHT_REGISTER["Default"]('guidance_in.mlp.2.weight', 'guidance_in.mlp.2.bias')

        self.weight_list = [
            self.img_in_proj,

            self.txt_in_input_embedder,
            self.txt_in_t_embedder_mlp_0,
            self.txt_in_t_embedder_mlp_2,
            self.txt_in_c_embedder_linear_1,
            self.txt_in_c_embedder_linear_2,

            self.txt_in_individual_token_refiner_blocks_0_norm1,
            self.txt_in_individual_token_refiner_blocks_0_self_attn_qkv,
            self.txt_in_individual_token_refiner_blocks_0_self_attn_proj,
            self.txt_in_individual_token_refiner_blocks_0_norm2,
            self.txt_in_individual_token_refiner_blocks_0_mlp_fc1,
            self.txt_in_individual_token_refiner_blocks_0_mlp_fc2,
            self.txt_in_individual_token_refiner_blocks_0_adaLN_modulation_1,

            self.txt_in_individual_token_refiner_blocks_1_norm1,
            self.txt_in_individual_token_refiner_blocks_1_self_attn_qkv,
            self.txt_in_individual_token_refiner_blocks_1_self_attn_proj,
            self.txt_in_individual_token_refiner_blocks_1_norm2,
            self.txt_in_individual_token_refiner_blocks_1_mlp_fc1,
            self.txt_in_individual_token_refiner_blocks_1_mlp_fc2,
            self.txt_in_individual_token_refiner_blocks_1_adaLN_modulation_1,

            self.time_in_mlp_0,
            self.time_in_mlp_2,
            self.vector_in_in_layer,
            self.vector_in_out_layer,
            self.guidance_in_mlp_0,
            self.guidance_in_mlp_2,
        ]

        for mm_weight in self.weight_list:
            if isinstance(mm_weight, MMWeightTemplate) or isinstance(mm_weight, LNWeightTemplate) or isinstance(mm_weight, Conv3dWeightTemplate):
                mm_weight.set_config(self.config['mm_config'])
                mm_weight.load(weight_dict)

    def to_cpu(self):
        for mm_weight in self.weight_list:
            if isinstance(mm_weight, MMWeightTemplate) or isinstance(mm_weight, LNWeightTemplate) or isinstance(mm_weight, Conv3dWeightTemplate):
                mm_weight.to_cpu()

    def to_cuda(self):
        for mm_weight in self.weight_list:
            if isinstance(mm_weight, MMWeightTemplate) or isinstance(mm_weight, LNWeightTemplate) or isinstance(mm_weight, Conv3dWeightTemplate):
                mm_weight.to_cuda()
