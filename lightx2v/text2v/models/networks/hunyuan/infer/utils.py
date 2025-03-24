import sgl_kernel


def rms_norm(x, weight, eps):
    x = x.contiguous()
    orig_shape = x.shape
    x = x.view(-1, orig_shape[-1])
    x = sgl_kernel.rmsnorm(x, weight, eps).view(orig_shape)
    return x
