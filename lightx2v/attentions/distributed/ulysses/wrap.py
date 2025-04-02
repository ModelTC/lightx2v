import functools
from lightx2v.attentions.distributed.ulysses.attn import ulysses_attn
from lightx2v.attentions.distributed.ring.attn import ring_attn

def parallelize_hunyuan(hunyuan_model):
    from lightx2v.attentions.distributed.utils.hunyuan.processor import pre_process, post_process
    """将 Hunyuan 模型的推理过程并行化，使用 Ulysses 注意力机制。

    参数:
        hunyuan_model: Hunyuan 模型实例，包含推理方法和其他属性。
    """
    # 将 Hunyuan 模型的并行注意力机制替换为 Ulysses 注意力
    hunyuan_model.transformer_infer.parallel_attention = ulysses_attn

    # 保存原始的推理方法，以便后续调用
    original_infer = hunyuan_model.infer

    @functools.wraps(hunyuan_model.__class__.infer)  # 保留原始推理方法的元信息
    def new_infer(self, text_encoders_output, image_encoder_output, args):
        """新的推理方法，处理输入并调用原始推理方法。

        参数:
            self: Hunyuan 模型实例
            text_encoders_output: 文本编码器的输出
            args: 其他参数

        返回:
            None
        """
        # 保存原始的潜在模型输入和频率数据
        self.scheduler.ori_latents, self.scheduler.ori_freqs_cos, self.scheduler.ori_freqs_sin = (
            self.scheduler.latents, 
            self.scheduler.freqs_cos, 
            self.scheduler.freqs_sin
        )
        
        # 预处理输入数据以适应并行计算
        self.scheduler.latents, self.scheduler.freqs_cos, self.scheduler.freqs_sin, split_dim = pre_process(
            self.scheduler.latents, self.scheduler.freqs_cos, self.scheduler.freqs_sin
        )

        # 调用原始推理方法，获取输出
        original_infer(
            text_encoders_output, image_encoder_output, args
        )

        # 对输出进行后处理
        self.scheduler.noise_pred = post_process(self.scheduler.noise_pred, split_dim)
        
        # 恢复原始的潜在模型输入和频率数据
        self.scheduler.latents, self.scheduler.freqs_cos, self.scheduler.freqs_sin = (
            self.scheduler.ori_latents, 
            self.scheduler.ori_freqs_cos, 
            self.scheduler.ori_freqs_sin
        )

        # return combined_output  # 返回处理后的输出（当前被注释掉）

    # 将新的推理方法绑定到 Hunyuan 模型实例
    new_infer = new_infer.__get__(hunyuan_model)
    hunyuan_model.infer = new_infer  # 替换原始推理方法


def parallelize_wan(wan_model):
    from lightx2v.attentions.distributed.utils.wan.processor import pre_process, post_process
    wan_model.transformer_infer.parallel_attention = ring_attn

    original_infer = wan_model.transformer_infer.infer

    @functools.wraps(wan_model.transformer_infer.__class__.infer)  # 保留原始推理方法的元信息
    def new_infer(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):

        x = pre_process(
            x
        )

        x = original_infer(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)

        x = post_process(
            x
        )

        return x

    new_infer = new_infer.__get__(wan_model.transformer_infer)
    wan_model.transformer_infer.infer = new_infer  # 替换原始推理方法