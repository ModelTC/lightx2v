from lightx2v.attentions.distributed.partial_heads_attn.attn import partial_heads_attn


def parallelize_hunyuan(hunyuan_model):
    hunyuan_model.transformer_infer.parallel_attention = partial_heads_attn
