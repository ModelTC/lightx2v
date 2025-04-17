from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER, LN_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER
from lightx2v.common.ops.mm.mm_weight import MMWeightTemplate
from lightx2v.common.ops.norm.layer_norm_weight import LNWeightTemplate
from lightx2v.common.ops.norm.rms_norm_weight import RMSWeightTemplate


class WanTransformerWeights:
    def __init__(self, config):
        self.blocks_num = config["num_layers"]
        self.task = config["task"]
        self.config = config
        if config["do_mm_calib"]:
            self.mm_type = "Calib"
        else:
            self.mm_type = config["mm_config"].get("mm_type", "Default") if config["mm_config"] else "Default"

    def load_weights(self, weight_dict):
        self.blocks_weights = [WanTransformerAttentionBlock(i, self.task, self.mm_type, self.config) for i in range(self.blocks_num)]
        for block in self.blocks_weights:
            block.load_weights(weight_dict)

    def to_cpu(self):
        for block in self.blocks_weights:
            block.to_cpu()

    def to_cuda(self):
        for block in self.blocks_weights:
            block.to_cuda()


class WanTransformerAttentionBlock:
    def __init__(self, block_index, task, mm_type, config):
        self.block_index = block_index
        self.mm_type = mm_type
        self.task = task
        self.config = config


    def load_weights(self, weight_dict):
        self.self_attn_q = MM_WEIGHT_REGISTER[self.mm_type](f"blocks.{self.block_index}.attn1.to_q.weight", f"blocks.{self.block_index}.attn1.to_q.bias")
        self.self_attn_k = MM_WEIGHT_REGISTER[self.mm_type](f"blocks.{self.block_index}.attn1.to_k.weight", f"blocks.{self.block_index}.attn1.to_k.bias")
        self.self_attn_v = MM_WEIGHT_REGISTER[self.mm_type](f"blocks.{self.block_index}.attn1.to_v.weight", f"blocks.{self.block_index}.attn1.to_v.bias")
        self.self_attn_o = MM_WEIGHT_REGISTER[self.mm_type](f"blocks.{self.block_index}.attn1.to_out.0.weight", f"blocks.{self.block_index}.attn1.to_out.0.bias")
        self.self_attn_norm_q = RMS_WEIGHT_REGISTER["sgl-kernel"](f"blocks.{self.block_index}.attn1.norm_q.weight")
        self.self_attn_norm_k = RMS_WEIGHT_REGISTER["sgl-kernel"](f"blocks.{self.block_index}.attn1.norm_k.weight")

        self.norm3 = LN_WEIGHT_REGISTER["Default"](f"blocks.{self.block_index}.norm2.weight", f"blocks.{self.block_index}.norm2.bias", eps=1e-6)
        
        self.cross_attn_q = MM_WEIGHT_REGISTER[self.mm_type](f"blocks.{self.block_index}.attn2.to_q.weight", f"blocks.{self.block_index}.attn2.to_q.bias")
        self.cross_attn_k = MM_WEIGHT_REGISTER[self.mm_type](f"blocks.{self.block_index}.attn2.to_k.weight", f"blocks.{self.block_index}.attn2.to_k.bias")
        self.cross_attn_v = MM_WEIGHT_REGISTER[self.mm_type](f"blocks.{self.block_index}.attn2.to_v.weight", f"blocks.{self.block_index}.attn2.to_v.bias")
        self.cross_attn_o = MM_WEIGHT_REGISTER[self.mm_type](f"blocks.{self.block_index}.attn2.to_out.0.weight", f"blocks.{self.block_index}.attn2.to_out.0.bias")
        self.cross_attn_norm_q = RMS_WEIGHT_REGISTER["sgl-kernel"](f"blocks.{self.block_index}.attn2.norm_q.weight")
        self.cross_attn_norm_k = RMS_WEIGHT_REGISTER["sgl-kernel"](f"blocks.{self.block_index}.attn2.norm_k.weight")

        self.ffn_0 = MM_WEIGHT_REGISTER[self.mm_type](f"blocks.{self.block_index}.ffn.net.0.proj.weight", f"blocks.{self.block_index}.ffn.net.0.proj.bias")
        self.ffn_2 = MM_WEIGHT_REGISTER[self.mm_type](f"blocks.{self.block_index}.ffn.net.2.weight", f"blocks.{self.block_index}.ffn.net.2.bias")
        
        
        self.modulation = weight_dict[f"blocks.{self.block_index}.scale_shift_table"]

        self.weight_list = [
            self.self_attn_q,
            self.self_attn_k,
            self.self_attn_v,
            self.self_attn_o,
            self.self_attn_norm_q,
            self.self_attn_norm_k,
            self.norm3,
            self.cross_attn_q,
            self.cross_attn_k,
            self.cross_attn_v,
            self.cross_attn_o,
            self.cross_attn_norm_q,
            self.cross_attn_norm_k,
            self.ffn_0,
            self.ffn_2,
        ]

        if self.task == "i2v":
            self.cross_attn_k_img = MM_WEIGHT_REGISTER[self.mm_type](f"blocks.{self.block_index}.attn2.add_k_proj.weight", f"blocks.{self.block_index}.attn2.add_k_proj.bias")
            self.cross_attn_v_img = MM_WEIGHT_REGISTER[self.mm_type](f"blocks.{self.block_index}.attn2.add_v_proj.weight", f"blocks.{self.block_index}.attn2.add_v_proj.bias")
            self.cross_attn_norm_k_img = RMS_WEIGHT_REGISTER["sgl-kernel"](f"blocks.{self.block_index}.attn2.norm_added_k.weight")
            
            self.weight_list.append(self.cross_attn_k_img)
            self.weight_list.append(self.cross_attn_v_img)
            self.weight_list.append(self.cross_attn_norm_k_img)

        for mm_weight in self.weight_list:
            if isinstance(mm_weight, (MMWeightTemplate, LNWeightTemplate, RMSWeightTemplate)):
                mm_weight.set_config(self.config["mm_config"])
                mm_weight.load(weight_dict)

    def to_cpu(self):
        for mm_weight in self.weight_list:
            if isinstance(mm_weight, (MMWeightTemplate, LNWeightTemplate, RMSWeightTemplate)):
                mm_weight.to_cpu()
        self.modulation = self.modulation.cpu()

    def to_cuda(self):
        for mm_weight in self.weight_list:
            if isinstance(mm_weight, (MMWeightTemplate, LNWeightTemplate, RMSWeightTemplate)):
                mm_weight.to_cuda()
        self.modulation = self.modulation.cuda()

    def to_cpu_sync(self):
        for mm_weight in self.weight_list:
            if isinstance(mm_weight, (MMWeightTemplate, LNWeightTemplate, RMSWeightTemplate)):
                mm_weight.to_cpu(non_blocking=True)
        self.modulation = self.modulation.to("cpu", non_blocking=True)

    def to_cuda_sync(self):
        for mm_weight in self.weight_list:
            if isinstance(mm_weight, (MMWeightTemplate, LNWeightTemplate, RMSWeightTemplate)):
                mm_weight.to_cuda(non_blocking=True)
        self.modulation = self.modulation.cuda(non_blocking=True)
