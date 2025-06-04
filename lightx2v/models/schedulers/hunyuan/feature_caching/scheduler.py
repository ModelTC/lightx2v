from ..scheduler import HunyuanScheduler
import torch


class HunyuanSchedulerTeaCaching(HunyuanScheduler):
    def __init__(self, config):
        super().__init__(config)

    def clear(self):
        # TODO: Transformer实例的缓存清理
        # if self.previous_residual is not None:
        #     self.previous_residual = self.previous_residual.cpu()
        # if self.previous_modulated_input is not None:
        #     self.previous_modulated_input = self.previous_modulated_input.cpu()

        # self.previous_modulated_input = None
        # self.previous_residual = None
        # torch.cuda.empty_cache()
        pass


class HunyuanSchedulerTaylorCaching(HunyuanScheduler):
    def __init__(self, config):
        super().__init__(config)
        pattern = [True, False, False, False]
        self.caching_records = (pattern * ((config.infer_steps + 3) // 4))[:config.infer_steps]

    def clear(self):
        # TODO: Transformer实例的缓存清理
        pass


class HunyuanSchedulerAdaCaching(HunyuanScheduler):
    def __init__(self, config):
        super().__init__(config)

    def clear(self):
        # TODO: Transformer实例的缓存清理
        pass


class HunyuanSchedulerCustomCaching(HunyuanScheduler):
    def __init__(self, config):
        super().__init__(config)

    def clear(self):
        # TODO: Transformer实例的缓存清理
        pass
