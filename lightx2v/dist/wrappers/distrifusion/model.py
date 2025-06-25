from lightx2v.models.networks.wan.model import WanModel
from lightx2v.dist.wrappers.distrifusion.infer import DistriFusionWanTransformerInferWrapper
from lightx2v.dist.wrappers.distrifusion.weights import DistriFusionWanTransformerWeightsWrapper


class DistriFusionWanModelWrapper:
    def __init__(self, model: WanModel, config):
        self.model = model
        self.config = config
        
        self._wrap_transformer(self.model, self.config)

    def __getattr__(self, name: str):
        if name in self.__dict__:
            return getattr(self, name)
        else:
            return getattr(self.model, name)
        
    def __delattr__(self, name: str):
        if name in self.__dict__:
            del self.__dict__[name]
        else:
            del self.model.__dict__[name]
            
    def _wrap_transformer(self, model, config):
        model.transformer_weights = DistriFusionWanTransformerWeightsWrapper(model.transformer_weights, config)
        model.transformer_infer = DistriFusionWanTransformerInferWrapper(model.transformer_infer, config)
        
    def infer(self, inputs, is_warmup=True):
        pass