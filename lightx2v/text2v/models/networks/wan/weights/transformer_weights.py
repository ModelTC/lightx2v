from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER
from lightx2v.common.ops.mm.mm_weight import MMWeightTemplate


class WanTransformerWeights:
    def __init__(self, config):
        self.blocks_num = config["num_layers"]
        self.task = config['task']
        if config['do_mm_calib']:
            self.mm_type = 'Calib'
        else:
            self.mm_type = config['mm_config'].get('mm_type', 'Default') if config['mm_config'] else 'Default'

    def load_weights(self, weight_dict):
        self.blocks_weights = [
            WanTransformerAttentionBlock(i, self.task, self.mm_type) for i in range(self.blocks_num)
        ]
        for block in self.blocks_weights:
            block.load_weights(weight_dict)

class WanTransformerAttentionBlock:
    def __init__(self, block_index, task, mm_type):
        self.block_index = block_index
        self.mm_type = mm_type
        self.task = task

    def load_weights(self, weight_dict):
        if self.task == 't2v':
            layers = {
                "self_attn_q": ["self_attn.q.weight", "self_attn.q.bias"],
                "self_attn_k": ["self_attn.k.weight", "self_attn.k.bias"],
                "self_attn_v": ["self_attn.v.weight", "self_attn.v.bias"],
                "self_attn_o": ["self_attn.o.weight", "self_attn.o.bias"],
                "self_attn_norm_q_weight": "self_attn.norm_q.weight",
                "self_attn_norm_k_weight": "self_attn.norm_k.weight",
                "norm3_weight": "norm3.weight",
                "norm3_bias": "norm3.bias",
                "cross_attn_q": ["cross_attn.q.weight", "cross_attn.q.bias"],
                "cross_attn_k": ["cross_attn.k.weight", "cross_attn.k.bias"],
                "cross_attn_v": ["cross_attn.v.weight", "cross_attn.v.bias"],
                "cross_attn_o": ["cross_attn.o.weight", "cross_attn.o.bias"],
                "cross_attn_norm_q_weight": "cross_attn.norm_q.weight",
                "cross_attn_norm_k_weight": "cross_attn.norm_k.weight",
                "ffn_0": ["ffn.0.weight", "ffn.0.bias"],
                "ffn_2": ["ffn.2.weight", "ffn.2.bias"],
                "modulation": "modulation",
            }
        elif self.task == 'i2v':
            layers = {
                "self_attn_q": ["self_attn.q.weight", "self_attn.q.bias"],
                "self_attn_k": ["self_attn.k.weight", "self_attn.k.bias"],
                "self_attn_v": ["self_attn.v.weight", "self_attn.v.bias"],
                "self_attn_o": ["self_attn.o.weight", "self_attn.o.bias"],
                "self_attn_norm_q_weight": "self_attn.norm_q.weight",
                "self_attn_norm_k_weight": "self_attn.norm_k.weight",
                "norm3_weight": "norm3.weight",
                "norm3_bias": "norm3.bias",
                "cross_attn_q": ["cross_attn.q.weight", "cross_attn.q.bias"],
                "cross_attn_k": ["cross_attn.k.weight", "cross_attn.k.bias"],
                "cross_attn_v": ["cross_attn.v.weight", "cross_attn.v.bias"],
                "cross_attn_o": ["cross_attn.o.weight", "cross_attn.o.bias"],
                "cross_attn_norm_q_weight": "cross_attn.norm_q.weight",
                "cross_attn_norm_k_weight": "cross_attn.norm_k.weight",
                "cross_attn_k_img": ["cross_attn.k_img.weight", "cross_attn.k_img.bias"],
                "cross_attn_v_img": ["cross_attn.v_img.weight", "cross_attn.v_img.bias"],
                "cross_attn_norm_k_img_weight": "cross_attn.norm_k_img.weight",
                "ffn_0": ["ffn.0.weight", "ffn.0.bias"],
                "ffn_2": ["ffn.2.weight", "ffn.2.bias"],
                "modulation": "modulation",
            }

        for layer_name, weight_keys in layers.items():
            if isinstance(weight_keys, list):
                weight_key, bias_key = weight_keys
                weight_path = f"blocks.{self.block_index}.{weight_key}"
                bias_path = f"blocks.{self.block_index}.{bias_key}"
                setattr(self, layer_name, MM_WEIGHT_REGISTER[self.mm_type](weight_path, bias_path))
                getattr(self, layer_name).load(weight_dict)
            else:
                weight_path = f"blocks.{self.block_index}.{weight_keys}"
                setattr(self, layer_name, weight_dict[weight_path])