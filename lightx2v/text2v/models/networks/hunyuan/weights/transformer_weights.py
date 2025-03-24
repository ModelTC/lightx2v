from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER
from lightx2v.common.ops.norm.rms_norm_weight import RMS_WEIGHT_REGISTER
from lightx2v.common.ops.mm.mm_weight import MMWeightTemplate
from lightx2v.common.ops.norm.rms_norm_weight import RMSWeightTemplate


class HunyuanTransformerWeights:
    def __init__(self, config):
        self.config = config
        self.init()

    def init(self):
        self.double_blocks_num = 20
        self.single_blocks_num = 40

    def load_weights(self, weight_dict):
        self.double_blocks_weights = [HunyuanTransformerDoubleBlock(i, self.config) for i in range(self.double_blocks_num)]
        self.single_blocks_weights = [HunyuanTransformerSingleBlock(i, self.config) for i in range(self.single_blocks_num)]
        for double_block in self.double_blocks_weights:
            double_block.load_weights(weight_dict)
        for single_block in self.single_blocks_weights:
            single_block.load_weights(weight_dict)

    def to_cpu(self):
        for double_block in self.double_blocks_weights:
            double_block.to_cpu()
        for single_block in self.single_blocks_weights:
            single_block.to_cpu()

    def to_cuda(self):
        for double_block in self.double_blocks_weights:
            double_block.to_cuda()
        for single_block in self.single_blocks_weights:
            single_block.to_cuda()


class HunyuanTransformerDoubleBlock:
    def __init__(self, block_index, config):
        self.block_index = block_index
        self.config = config
        self.weight_list = []

    def load_weights(self, weight_dict):
        if self.config['do_mm_calib']:
            mm_type = 'Calib'
        else:
            mm_type = self.config['mm_config'].get('mm_type', 'Default') if self.config['mm_config'] else 'Default'

        self.img_mod = MM_WEIGHT_REGISTER[mm_type](f'double_blocks.{self.block_index}.img_mod.linear.weight', f'double_blocks.{self.block_index}.img_mod.linear.bias')
        self.img_attn_qkv = MM_WEIGHT_REGISTER[mm_type](f'double_blocks.{self.block_index}.img_attn_qkv.weight', f'double_blocks.{self.block_index}.img_attn_qkv.bias')
        self.img_attn_q_norm = RMS_WEIGHT_REGISTER['sgl-kernel'](f'double_blocks.{self.block_index}.img_attn_q_norm.weight', eps=1e-6)
        self.img_attn_k_norm = RMS_WEIGHT_REGISTER['sgl-kernel'](f'double_blocks.{self.block_index}.img_attn_k_norm.weight', eps=1e-6)
        self.img_attn_proj = MM_WEIGHT_REGISTER[mm_type](f'double_blocks.{self.block_index}.img_attn_proj.weight', f'double_blocks.{self.block_index}.img_attn_proj.bias')
        self.img_mlp_fc1 = MM_WEIGHT_REGISTER[mm_type](f'double_blocks.{self.block_index}.img_mlp.fc1.weight', f'double_blocks.{self.block_index}.img_mlp.fc1.bias')
        self.img_mlp_fc2 = MM_WEIGHT_REGISTER[mm_type](f'double_blocks.{self.block_index}.img_mlp.fc2.weight', f'double_blocks.{self.block_index}.img_mlp.fc2.bias')

        self.txt_mod = MM_WEIGHT_REGISTER[mm_type](f'double_blocks.{self.block_index}.txt_mod.linear.weight', f'double_blocks.{self.block_index}.txt_mod.linear.bias')
        self.txt_attn_qkv = MM_WEIGHT_REGISTER[mm_type](f'double_blocks.{self.block_index}.txt_attn_qkv.weight', f'double_blocks.{self.block_index}.txt_attn_qkv.bias')
        self.txt_attn_q_norm = RMS_WEIGHT_REGISTER['sgl-kernel'](f'double_blocks.{self.block_index}.txt_attn_q_norm.weight', eps=1e-6)
        self.txt_attn_k_norm = RMS_WEIGHT_REGISTER['sgl-kernel'](f'double_blocks.{self.block_index}.txt_attn_k_norm.weight', eps=1e-6)
        self.txt_attn_proj = MM_WEIGHT_REGISTER[mm_type](f'double_blocks.{self.block_index}.txt_attn_proj.weight', f'double_blocks.{self.block_index}.txt_attn_proj.bias')
        self.txt_mlp_fc1 = MM_WEIGHT_REGISTER[mm_type](f'double_blocks.{self.block_index}.txt_mlp.fc1.weight', f'double_blocks.{self.block_index}.txt_mlp.fc1.bias')
        self.txt_mlp_fc2 = MM_WEIGHT_REGISTER[mm_type](f'double_blocks.{self.block_index}.txt_mlp.fc2.weight', f'double_blocks.{self.block_index}.txt_mlp.fc2.bias')

        self.weight_list = [
            self.img_mod,
            self.img_attn_qkv,
            self.img_attn_q_norm,
            self.img_attn_k_norm,
            self.img_attn_proj,
            self.img_mlp_fc1,
            self.img_mlp_fc2,
            self.txt_mod,
            self.txt_attn_qkv,
            self.txt_attn_q_norm,
            self.txt_attn_k_norm,
            self.txt_attn_proj,
            self.txt_mlp_fc1,
            self.txt_mlp_fc2,
        ]

        for mm_weight in self.weight_list:
            if isinstance(mm_weight, MMWeightTemplate) or isinstance(mm_weight, RMSWeightTemplate):
                mm_weight.set_config(self.config['mm_config'])
                mm_weight.load(weight_dict)

    def to_cpu(self):
        for mm_weight in self.weight_list:
            if isinstance(mm_weight, MMWeightTemplate) or isinstance(mm_weight, RMSWeightTemplate):
                mm_weight.to_cpu()

    def to_cuda(self):
        for mm_weight in self.weight_list:
            if isinstance(mm_weight, MMWeightTemplate) or isinstance(mm_weight, RMSWeightTemplate):
                mm_weight.to_cuda()


class HunyuanTransformerSingleBlock:
    def __init__(self, block_index, config):
        self.block_index = block_index
        self.config = config
        self.weight_list = []

    def load_weights(self, weight_dict):
        if self.config['do_mm_calib']:
            mm_type = 'Calib'
        else:
            mm_type = self.config['mm_config'].get('mm_type', 'Default') if self.config['mm_config'] else 'Default'

        self.linear1 = MM_WEIGHT_REGISTER[mm_type](f'single_blocks.{self.block_index}.linear1.weight', f'single_blocks.{self.block_index}.linear1.bias')
        self.linear2 = MM_WEIGHT_REGISTER[mm_type](f'single_blocks.{self.block_index}.linear2.weight', f'single_blocks.{self.block_index}.linear2.bias')
        self.q_norm = RMS_WEIGHT_REGISTER['sgl-kernel'](f'single_blocks.{self.block_index}.q_norm.weight', eps=1e-6)
        self.k_norm = RMS_WEIGHT_REGISTER['sgl-kernel'](f'single_blocks.{self.block_index}.k_norm.weight', eps=1e-6)
        self.modulation = MM_WEIGHT_REGISTER[mm_type](f'single_blocks.{self.block_index}.modulation.linear.weight', f'single_blocks.{self.block_index}.modulation.linear.bias')

        self.weight_list = [
            self.linear1,
            self.linear2,
            self.q_norm,
            self.k_norm,
            self.modulation,
        ]

        for mm_weight in self.weight_list:
            if isinstance(mm_weight, MMWeightTemplate) or isinstance(mm_weight, RMSWeightTemplate):
                mm_weight.set_config(self.config['mm_config'])
                mm_weight.load(weight_dict)

    def to_cpu(self):
        for mm_weight in self.weight_list:
            if isinstance(mm_weight, MMWeightTemplate) or isinstance(mm_weight, RMSWeightTemplate):
                mm_weight.to_cpu()

    def to_cuda(self):
        for mm_weight in self.weight_list:
            if isinstance(mm_weight, MMWeightTemplate) or isinstance(mm_weight, RMSWeightTemplate):
                mm_weight.to_cuda()
