import torch
import sgl_kernel
import torch.cuda.amp as amp


def rms_norm(x, weight, eps):
    x = x.contiguous()
    orig_shape = x.shape
    x = x.view(-1, orig_shape[-1])
    x = sgl_kernel.rmsnorm(x, weight, eps).view(orig_shape)
    return x


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


def apply_rotary_emb(x, freqs_i):
    n = x.size(1)
    seq_len = freqs_i.size(0)

    x_i = torch.view_as_complex(
        x[:seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
    )
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
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1).to(torch.bfloat16)
    return x
