import functools
from lightx2v.attentions.distributed.ring.attn import ring_attn
from lightx2v.attentions.distributed.utils.process import pre_process, post_process


def parallelize_hunyuan(hunyuan_model):
    """将 Hunyuan 模型的推理过程并行化，使用 Ulysses 注意力机制。

    参数:
        hunyuan_model: Hunyuan 模型实例，包含推理方法和其他属性。
    """
    # 将 Hunyuan 模型的并行注意力机制替换为 Ulysses 注意力
    hunyuan_model.transformer_infer.parallel_attention = ring_attn

    # 保存原始的推理方法，以便后续调用
    original_infer = hunyuan_model.infer

    @functools.wraps(hunyuan_model.__class__.infer)  # 保留原始推理方法的元信息
    def new_infer(
        self,
        latent_model_input,
        t_expand,
        text_states,
        text_mask,
        text_states_2,
        freqs_cos,
        freqs_sin,
        guidance,
    ):
        """新的推理方法，处理输入并调用原始推理方法。

        参数:
            self: Hunyuan 模型实例
            latent_model_input: 潜在模型输入
            t_expand: 时间扩展参数
            text_states: 文本状态
            text_mask: 文本掩码
            text_states_2: 第二组文本状态
            freqs_cos: 余弦频率
            freqs_sin: 正弦频率
            guidance: 指导参数

        返回:
            combined_output: 经过后处理的输出结果
        """
        # 预处理输入数据
        latent_model_input, freqs_cos, freqs_sin, split_dim = pre_process(
            latent_model_input, freqs_cos, freqs_sin
        )

        # 调用原始推理方法，获取输出
        output = original_infer(
            latent_model_input,
            t_expand,
            text_states,
            text_mask,
            text_states_2,
            freqs_cos,
            freqs_sin,
            guidance,
        )

        # 对输出进行后处理
        combined_output = post_process(output, split_dim)

        return combined_output  # 返回处理后的输出

    # 将新的推理方法绑定到 Hunyuan 模型实例
    new_infer = new_infer.__get__(hunyuan_model)
    hunyuan_model.infer = new_infer  # 替换原始推理方法
