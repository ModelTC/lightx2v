import torch.distributed as dist
from lightx2v.models.networks.wan.weights.transformer_weights import (
    WanTransformerWeights,
)


class DistriFusionWanTransformerWeightsWrapper:
    def __init__(self, transformer_weights: WanTransformerWeights, config):
        pass