import torch
import math
from .utils import rope_params, sinusoidal_embedding_1d
import torch.cuda.amp as amp
from loguru import logger

class WanPreInfer:
    def __init__(self, config):
        assert (config["dim"] % config["num_heads"]) == 0 and (config["dim"] // config["num_heads"]) % 2 == 0
        d = config["dim"] // config["num_heads"]

        self.task = config["task"]
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

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def infer(self, weights, inputs, positive, kv_start=0, kv_end=0, overlap=0):
        x = [self.scheduler.latents]

        if self.scheduler.flag_df:
            t = self.scheduler.df_timesteps[self.scheduler.step_index].unsqueeze(0)
            assert t.dim() == 2  # df推理模型timestep是二维
        else:
            t = torch.stack([self.scheduler.timesteps[self.scheduler.step_index]])

        if positive:
            context = inputs["text_encoder_output"]["context"]
        else:
            context = inputs["text_encoder_output"]["context_null"]
        seq_len = self.scheduler.seq_len

        if self.task == "i2v":
            clip_fea = inputs["image_encoder_output"]["clip_encoder_out"]

            image_encoder = inputs["image_encoder_output"]["vae_encode_out"]
                
            y = [image_encoder]
            logger.info(f"y[0]:{y[0].shape}, x[0]:{x[0].shape}, image_encoder:{image_encoder.shape}")
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [weights.patch_embedding.apply(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long).cuda()
        assert seq_lens.max() <= seq_len
        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

        embed = sinusoidal_embedding_1d(self.freq_dim, t.flatten())
        embed = weights.time_embedding_0.apply(embed)
        embed = torch.nn.functional.silu(embed)
        embed = weights.time_embedding_2.apply(embed)
        embed0 = torch.nn.functional.silu(embed)

        embed0 = weights.time_projection_1.apply(embed0).unflatten(1, (6, self.dim))

        if self.scheduler.flag_df:
            b, f = t.shape
            assert b == len(x)  # batch_size == 1
            embed = embed.view(b, f, 1, 1, self.dim)
            embed0 = embed0.view(b, f, 1, 1, 6, self.dim)
            embed = embed.repeat(1, 1, grid_sizes[0][1], grid_sizes[0][2], 1).flatten(1, 3)
            embed0 = embed0.repeat(1, 1, grid_sizes[0][1], grid_sizes[0][2], 1, 1).flatten(1, 3)
            embed0 = embed0.transpose(1, 2).contiguous()

        # text embeddings
        stacked = torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
        out = weights.text_embedding_0.apply(stacked.squeeze(0))
        out = torch.nn.functional.gelu(out, approximate="tanh")
        context = weights.text_embedding_2.apply(out)

        if self.task == "i2v":
            context_clip = weights.proj_0.apply(clip_fea)
            context_clip = weights.proj_1.apply(context_clip)
            context_clip = torch.nn.functional.gelu(context_clip, approximate="none")
            context_clip = weights.proj_3.apply(context_clip)
            context_clip = weights.proj_4.apply(context_clip)

            context = torch.concat([context_clip, context], dim=0)

        return (
            embed,
            grid_sizes,
            (x.squeeze(0), embed0.squeeze(0), seq_lens, self.freqs, context),
        )
