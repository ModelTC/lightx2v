import torch
import torch.distributed as dist


def compute_freqs(c, grid_sizes, freqs):
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    f, h, w = grid_sizes[0].tolist()
    seq_len = f * h * w
    freqs_i = torch.cat(
        [
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(seq_len, 1, -1)

    return freqs_i


def compute_freqs_causvid(c, grid_sizes, freqs, start_frame=0):
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    f, h, w = grid_sizes[0].tolist()
    seq_len = f * h * w
    freqs_i = torch.cat(
        [
            freqs[0][start_frame : start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(seq_len, 1, -1)

    return freqs_i


def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(pad_size, s1, s2, dtype=original_tensor.dtype, device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor


def compute_freqs_dist(s, c, grid_sizes, freqs):
    world_size = dist.get_world_size()
    cur_rank = dist.get_rank()
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    f, h, w = grid_sizes[0].tolist()
    seq_len = f * h * w
    freqs_i = torch.cat(
        [
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(seq_len, 1, -1)

    freqs_i = pad_freqs(freqs_i, s * world_size)
    s_per_rank = s
    freqs_i_rank = freqs_i[(cur_rank * s_per_rank) : ((cur_rank + 1) * s_per_rank), :, :]
    return freqs_i_rank


def apply_rotary_emb(x, freqs_i):
    n = x.size(1)
    seq_len = freqs_i.size(0)

    x_i = torch.view_as_complex(x[:seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
    # Apply rotary embedding
    x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
    x_i = torch.cat([x_i, x[seq_len:]]).to(torch.bfloat16)
    return x_i


def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1).to(torch.bfloat16)
    return x
