import torch


class CogvideoxPostInfer:
    def __init__(self, config):
        self.config = config

    def ada_layernorm(self, weight_mm, weight_ln, x, temb):
        temb = torch.nn.functional.silu(temb)
        temb = weight_mm.apply(temb)
        shift, scale = temb.chunk(2, dim=1)
        x = weight_ln.apply(x) * (1 + scale) + shift
        return x

    def infer(self, weight, hidden_states, encoder_hidden_states, temb, infer_shapes):
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=0)
        hidden_states = weight.norm_final.apply(hidden_states)
        hidden_states = hidden_states[self.config.text_len :,]
        hidden_states = self.ada_layernorm(weight.norm_out_linear, weight.norm_out_norm, hidden_states, temb=temb)
        hidden_states = weight.proj_out.apply(hidden_states)
        p = self.config["patch_size"]
        p_t = self.config["patch_size_t"]
        num_frames, _, height, width = infer_shapes
        output = hidden_states.reshape((num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p)
        output = output.permute(0, 4, 3, 1, 5, 2, 6).flatten(5, 6).flatten(3, 4).flatten(0, 1)
        return output
