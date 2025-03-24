class WanPostWeights:
    def __init__(self):
        pass

    def load_weights(self, weight_dict):
        head_layers = {"head": ["head.weight", "head.bias", "modulation"]}
        for param_name, param_keys in head_layers.items():
            for key in param_keys:
                weight_path = f"{param_name}.{key}"
                key = key.split('.')
                setattr(self, f"{param_name}_{key[-1]}", weight_dict[weight_path])