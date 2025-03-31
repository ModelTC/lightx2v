from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER, LN_WEIGHT_REGISTER
from lightx2v.common.ops.mm.mm_weight import MMWeightTemplate
from lightx2v.common.ops.norm.layer_norm_weight import LNWeightTemplate


class WanTransformerWeights:
    def __init__(self, config):
        self.blocks_num = config["num_layers"]
        self.task = config["task"]
        self.config = config
        if config["do_mm_calib"]:
            self.mm_type = "Calib"
        else:
            self.mm_type = (
                config["mm_config"].get("mm_type", "Default") if config["mm_config"] else "Default"
            )

    def load_weights(self, weight_dict):
        self.blocks_weights = [
            WanTransformerAttentionBlock(i, self.task, self.mm_type, self.config)
            for i in range(self.blocks_num)
        ]
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
        self.self_attn_q = MM_WEIGHT_REGISTER[self.mm_type](
            f"blocks.{self.block_index}.self_attn.q.weight",
            f"blocks.{self.block_index}.self_attn.q.bias",
        )
        self.self_attn_k = MM_WEIGHT_REGISTER[self.mm_type](
            f"blocks.{self.block_index}.self_attn.k.weight",
            f"blocks.{self.block_index}.self_attn.k.bias",
        )
        self.self_attn_v = MM_WEIGHT_REGISTER[self.mm_type](
            f"blocks.{self.block_index}.self_attn.v.weight",
            f"blocks.{self.block_index}.self_attn.v.bias",
        )
        self.self_attn_o = MM_WEIGHT_REGISTER[self.mm_type](
            f"blocks.{self.block_index}.self_attn.o.weight",
            f"blocks.{self.block_index}.self_attn.o.bias",
        )
        self.self_attn_norm_q_weight = weight_dict[
            f"blocks.{self.block_index}.self_attn.norm_q.weight"
        ]
        self.self_attn_norm_k_weight = weight_dict[
            f"blocks.{self.block_index}.self_attn.norm_k.weight"
        ]
        self.norm3 = LN_WEIGHT_REGISTER[self.mm_type](
            f"blocks.{self.block_index}.norm3.weight",
            f"blocks.{self.block_index}.norm3.bias",
            eps=1e-6,
        )
        self.cross_attn_q = MM_WEIGHT_REGISTER[self.mm_type](
            f"blocks.{self.block_index}.cross_attn.q.weight",
            f"blocks.{self.block_index}.cross_attn.q.bias",
        )
        self.cross_attn_k = MM_WEIGHT_REGISTER[self.mm_type](
            f"blocks.{self.block_index}.cross_attn.k.weight",
            f"blocks.{self.block_index}.cross_attn.k.bias",
        )
        self.cross_attn_v = MM_WEIGHT_REGISTER[self.mm_type](
            f"blocks.{self.block_index}.cross_attn.v.weight",
            f"blocks.{self.block_index}.cross_attn.v.bias",
        )
        self.cross_attn_o = MM_WEIGHT_REGISTER[self.mm_type](
            f"blocks.{self.block_index}.cross_attn.o.weight",
            f"blocks.{self.block_index}.cross_attn.o.bias",
        )
        self.cross_attn_norm_q_weight = weight_dict[
            f"blocks.{self.block_index}.cross_attn.norm_q.weight"
        ]
        self.cross_attn_norm_k_weight = weight_dict[
            f"blocks.{self.block_index}.cross_attn.norm_k.weight"
        ]
        self.ffn_0 = MM_WEIGHT_REGISTER[self.mm_type](
            f"blocks.{self.block_index}.ffn.0.weight", f"blocks.{self.block_index}.ffn.0.bias"
        )
        self.ffn_2 = MM_WEIGHT_REGISTER[self.mm_type](
            f"blocks.{self.block_index}.ffn.2.weight", f"blocks.{self.block_index}.ffn.2.bias"
        )
        self.modulation = weight_dict[f"blocks.{self.block_index}.modulation"]

        self.weight_list = [
            self.self_attn_q,
            self.self_attn_k,
            self.self_attn_v,
            self.self_attn_o,
            self.self_attn_norm_q_weight,
            self.self_attn_norm_k_weight,
            self.norm3,
            self.cross_attn_q,
            self.cross_attn_k,
            self.cross_attn_v,
            self.cross_attn_o,
            self.cross_attn_norm_q_weight,
            self.cross_attn_norm_k_weight,
            self.ffn_0,
            self.ffn_2,
            self.modulation,
        ]

        if self.task == "i2v":
            self.cross_attn_k_img = MM_WEIGHT_REGISTER[self.mm_type](
                f"blocks.{self.block_index}.cross_attn.k_img.weight",
                f"blocks.{self.block_index}.cross_attn.k_img.bias",
            )
            self.cross_attn_v_img = MM_WEIGHT_REGISTER[self.mm_type](
                f"blocks.{self.block_index}.cross_attn.v_img.weight",
                f"blocks.{self.block_index}.cross_attn.v_img.bias",
            )
            self.cross_attn_norm_k_img_weight = weight_dict[
                f"blocks.{self.block_index}.cross_attn.norm_k_img.weight"
            ]
            self.weight_list.append(self.cross_attn_k_img)
            self.weight_list.append(self.cross_attn_v_img)
            self.weight_list.append(self.cross_attn_norm_k_img_weight)

        for mm_weight in self.weight_list:
            if isinstance(mm_weight, MMWeightTemplate) or isinstance(mm_weight, LNWeightTemplate):
                mm_weight.set_config(self.config["mm_config"])
                mm_weight.load(weight_dict)

    def to_cpu(self):
        for mm_weight in self.weight_list:
            if isinstance(mm_weight, MMWeightTemplate) or isinstance(mm_weight, LNWeightTemplate):
                mm_weight.to_cpu()
            else:
                mm_weight.cpu()

    def to_cuda(self):
        for mm_weight in self.weight_list:
            if isinstance(mm_weight, MMWeightTemplate) or isinstance(mm_weight, LNWeightTemplate):
                mm_weight.to_cuda()
            else:
                mm_weight.cuda()
