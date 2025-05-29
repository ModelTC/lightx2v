import torch
import torch.nn.functional as F


def apply_rotary_emb(x, freqs_cis, use_real=True, use_real_unbind_dim=-1):
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None]
        sin = sin[None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        # used for lumina
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(2)

        return x_out.type_as(x)


class CogvideoxTransformerInfer:
    def __init__(self, config):
        self.config = config
        self.attn_type = "torch_sdpa"

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def infer(self, weights, hidden_states, encoder_hidden_states, temb):
        image_rotary_emb = self.scheduler.image_rotary_emb
        for i in range(self.config.transformer_num_layers):
            hidden_states, encoder_hidden_states = self.infer_block(
                weights.blocks_weights[i],
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

        return hidden_states, encoder_hidden_states

    def cogvideox_norm1(self, weights, hidden_states, encoder_hidden_states, temb):
        temb = torch.nn.functional.silu(temb)
        temb = weights.norm1_linear.apply(temb)
        shift, scale, gate, enc_shift, enc_scale, enc_gate = temb.chunk(6, dim=1)
        hidden_states = weights.norm1_norm.apply(hidden_states) * (1 + scale)[:, :] + shift[:, :]
        encoder_hidden_states = weights.norm1_norm.apply(encoder_hidden_states) * (1 + enc_scale)[:, :] + enc_shift[:, :]
        return hidden_states, encoder_hidden_states, gate, enc_gate

    def cogvideox_norm2(self, weights, hidden_states, encoder_hidden_states, temb):
        temb = torch.nn.functional.silu(temb)
        temb = weights.norm2_linear.apply(temb)
        shift, scale, gate, enc_shift, enc_scale, enc_gate = temb.chunk(6, dim=1)
        hidden_states = weights.norm2_norm.apply(hidden_states) * (1 + scale)[:, :] + shift[:, :]
        encoder_hidden_states = weights.norm2_norm.apply(encoder_hidden_states) * (1 + enc_scale)[:, :] + enc_shift[:, :]
        return hidden_states, encoder_hidden_states, gate, enc_gate

    def cogvideox_attention(self, weights, hidden_states, encoder_hidden_states, image_rotary_emb):
        text_seq_length = encoder_hidden_states.size(0)
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=0)

        query = weights.attn1_to_q.apply(hidden_states)
        key = weights.attn1_to_k.apply(hidden_states)
        value = weights.attn1_to_v.apply(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.config.transformer_num_attention_heads

        query = query.view(-1, self.config.transformer_num_attention_heads, head_dim).transpose(0, 1)
        key = key.view(-1, self.config.transformer_num_attention_heads, head_dim).transpose(0, 1)
        value = value.view(-1, self.config.transformer_num_attention_heads, head_dim).transpose(0, 1)

        query = weights.attn1_norm_q.apply(query)
        key = weights.attn1_norm_k.apply(key)

        query[:, text_seq_length:] = apply_rotary_emb(query[:, text_seq_length:], image_rotary_emb)
        key[:, text_seq_length:] = apply_rotary_emb(key[:, text_seq_length:], image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(query[None], key[None], value[None], attn_mask=None, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(1, -1, self.config.transformer_num_attention_heads * head_dim)
        hidden_states = hidden_states.squeeze(0)

        hidden_states = weights.attn1_to_out.apply(hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split([text_seq_length, hidden_states.size(0) - text_seq_length], dim=0)
        return hidden_states, encoder_hidden_states

    def cogvideox_ff(self, weights, hidden_states):
        hidden_states = weights.ff_net_0_proj.apply(hidden_states)
        hidden_states = torch.nn.functional.gelu(hidden_states, approximate="tanh")
        hidden_states = weights.ff_net_2_proj.apply(hidden_states)
        return hidden_states

    @torch.no_grad()
    def infer_block(self, weights, hidden_states, encoder_hidden_states, temb, image_rotary_emb):
        text_seq_length = encoder_hidden_states.size(0)

        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.cogvideox_norm1(weights, hidden_states, encoder_hidden_states, temb)

        attn_hidden_states, attn_encoder_hidden_states = self.cogvideox_attention(
            weights,
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.cogvideox_norm2(weights, hidden_states, encoder_hidden_states, temb)

        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=0)
        ff_output = self.cogvideox_ff(weights, norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output[text_seq_length:,]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:text_seq_length,]

        return hidden_states, encoder_hidden_states
