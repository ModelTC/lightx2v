import torch


class WanPreWeights:
    def __init__(self, config):
        self.in_dim = config["in_dim"]
        self.dim = config["dim"]
        self.patch_size = (1, 2, 2)

    def load_weights(self, weight_dict):
        layers = {
            "text_embedding": {"0": ["weight", "bias"], "2": ["weight", "bias"]},
            "time_embedding": {"0": ["weight", "bias"], "2": ["weight", "bias"]},
            "time_projection": {"1": ["weight", "bias"]},
        }

        self.patch_embedding = (
            torch.nn.Conv3d(
                self.in_dim,
                self.dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )
            .to(torch.bfloat16)
            .cuda()
        )
        self.patch_embedding.weight.data.copy_(weight_dict["patch_embedding.weight"])
        self.patch_embedding.bias.data.copy_(weight_dict["patch_embedding.bias"])
        for module_name, sub_layers in layers.items():
            for param_name, param_keys in sub_layers.items():
                for key in param_keys:
                    weight_path = f"{module_name}.{param_name}.{key}"
                    setattr(
                        self,
                        f"{module_name}_{param_name}_{key}",
                        weight_dict[weight_path],
                    )

        if 'img_emb.proj.0.weight' in weight_dict.keys():
            MLP_layers = {
                "proj_0_weight": "proj.0.weight",
                "proj_0_bias": "proj.0.bias",
                "proj_1_weight": "proj.1.weight",
                "proj_1_bias": "proj.1.bias",
                "proj_3_weight": "proj.3.weight",
                "proj_3_bias": "proj.3.bias",
                "proj_4_weight": "proj.4.weight",
                "proj_4_bias": "proj.4.bias",
            }

            for layer_name, weight_keys in MLP_layers.items():
                weight_path = f"img_emb.{weight_keys}"
                setattr(self, layer_name, weight_dict[weight_path])