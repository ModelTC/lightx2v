import torch
import torch.nn as nn


class MemoryEfficientBlocks(nn.Module):
    def __init__(self, block_class, num_blocks, **block_params):
        super().__init__()
        self.block_class = block_class
        self.num_blocks = num_blocks
        self.block_params = block_params

        # 初始化两个block
        self.active_blocks = nn.ModuleList([block_class(**block_params) for _ in range(2)])

        # 为权重加载创建独立的CUDA流，并设置优先级
        self.compute_stream = torch.cuda.Stream(priority=-1)  # 高优先级
        self.load_stream = torch.cuda.Stream(priority=0)  # 普通优先级

        # 预分配固定内存用于异步传输
        self.pinned_memory = torch.cuda.empty_cache()
        torch.cuda.memory.set_per_process_memory_fraction(0.8)  # 限制GPU内存使用

        # 用于存储预加载的权重
        # self.next_weights = None
        self.weight_buffer = []
        # self.current_block_idx = 0

    def initialize_weights(self, checkpoint, key):
        """加载所有权重到CPU内存"""
        # checkpoint = torch.load(checkpoint_path, map_location='cpu')
        for i in range(self.num_blocks):
            block_weights = {k.replace(f"{key}.{i}.", ""): v for k, v in checkpoint.items() if f"{key}.{i}." in k}
            self.weight_buffer.append(block_weights)

    def prefetch_weights(self, block_idx):
        """在独立CUDA流中预加载下一个block的权重"""
        with torch.cuda.stream(self.load_stream):
            next_weights = self.weight_buffer[block_idx]
            next_weights = {k: v.cuda(non_blocking=True) for k, v in next_weights.items()}
            self.active_blocks[1].load_state_dict(next_weights)

    def swap_blocks(self):
        """交换两个block并更新权重"""
        # 等待计算完成
        self.compute_stream.synchronize()
        # 等待加载完成
        self.load_stream.synchronize()

        # 交换blocks
        self.active_blocks[0], self.active_blocks[1] = self.active_blocks[1], self.active_blocks[0]

    def forward(self, *args, **kwargs):
        """前向传播，同时进行计算和权重加载"""
        # import pdb; pdb.set_trace()
        for i in range(self.num_blocks):
            if i == 0:
                self.active_blocks[0].load_state_dict(self.weight_buffer[0])

            # 在主计算流中进行当前block的计算
            with torch.cuda.stream(self.compute_stream):
                current_block = self.active_blocks[0]
                outputs = current_block(*args, **kwargs)  # 解包参数传入
            # import pdb; pdb.set_trace()

            # 在独立流中预加载下一个block的权重
            if i < self.num_blocks - 1:
                self.prefetch_weights(i + 1)

            # 交换blocks并更新权重
            self.swap_blocks()

            # 更新args中的输入为当前输出
            args = list(args)
            if len(outputs) == 1:
                args[0] = outputs
            else:
                for i in range(len(outputs)):
                    args[i] = outputs[i]
            args = tuple(args)

        return outputs
