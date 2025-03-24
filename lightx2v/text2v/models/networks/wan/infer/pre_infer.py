import torch
import math
from .utils import rope_params, sinusoidal_embedding_1d
import torch.cuda.amp as amp


class WanPreInfer:
    def __init__(self, config):
        assert (config["dim"] % config["num_heads"]) == 0 and (
            config["dim"] // config["num_heads"]
        ) % 2 == 0
        d = config["dim"] // config["num_heads"]
        
        self.task = config['task']
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        ).cuda()
        self.freq_dim = config["freq_dim"]
        self.dim = config["dim"]
        self.text_len = config["text_len"]

    def infer(self, weights, x, t, context, seq_len, clip_fea=None, y=None):
        
        if self.task == 'i2v':
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [weights.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long).cuda()
        assert seq_lens.max() <= seq_len
        x = torch.cat(
            [
                torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
                for u in x
            ]
        )

        embed = sinusoidal_embedding_1d(self.freq_dim, t)
        embed = torch.addmm(
            weights.time_embedding_0_bias,
            embed,
            weights.time_embedding_0_weight.t(),
        )
        embed = torch.nn.functional.silu(embed)
        embed = torch.addmm(
            weights.time_embedding_2_bias,
            embed,
            weights.time_embedding_2_weight.t(),
        )
        embed0 = torch.nn.functional.silu(embed)

        embed0 = torch.addmm(
            weights.time_projection_1_bias,
            embed0,
            weights.time_projection_1_weight.t(),
        ).unflatten(1, (6, self.dim))

        # text embeddings
        stacked = torch.stack(
            [
                torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]
        )
        out = torch.addmm(
            weights.text_embedding_0_bias,
            stacked.squeeze(0),
            weights.text_embedding_0_weight.t(),
        )
        out = torch.nn.functional.gelu(out, approximate="tanh")
        context = torch.addmm(
            weights.text_embedding_2_bias,
            out,
            weights.text_embedding_2_weight.t(),
        )

        if self.task == 'i2v':
            context_clip = torch.nn.functional.layer_norm(
                clip_fea,
                normalized_shape=(clip_fea.shape[1],),
                weight=weights.proj_0_weight,
                bias=weights.proj_0_bias,
                eps=1e-5,
            )
            context_clip = torch.addmm(
                weights.proj_1_bias,
                context_clip,
                weights.proj_1_weight.t(),
            )
            context_clip = torch.nn.functional.gelu(context_clip, approximate="none")
            context_clip = torch.addmm(
                weights.proj_3_bias,
                context_clip,
                weights.proj_3_weight.t(),
            )
            context_clip = torch.nn.functional.layer_norm(
                context_clip,
                normalized_shape=(context_clip.shape[1],),
                weight=weights.proj_4_weight,
                bias=weights.proj_4_bias,
                eps=1e-5,
            )

            context = torch.concat([context_clip, context], dim=0)
            
        return (
            embed,
            grid_sizes,
            (x.squeeze(0), embed0.squeeze(0), seq_lens, self.freqs, context),
        )
