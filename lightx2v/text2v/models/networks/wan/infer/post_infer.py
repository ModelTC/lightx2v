import math
import torch
import torch.cuda.amp as amp


class WanPostInfer:
    def __init__(self, config):
        self.out_dim = config["out_dim"]
        self.patch_size = (1, 2, 2)

    def infer(self, weights, x, e, grid_sizes):
        e = (weights.head_modulation + e.unsqueeze(1)).chunk(2, dim=1)
        norm_out = torch.nn.functional.layer_norm(
            x, (x.shape[1],), None, None, 1e-6
        ).type_as(x)
        out = norm_out * (1 + e[1].squeeze(0)) + e[0].squeeze(0)
        x = weights.head.apply(out)
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        x = x.unsqueeze(0)
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out
