from lightx2v.models.schedulers.wan.scheduler import WanScheduler


# 1. TeaCaching: 对单步去噪的前半段，后半段分别缓存
class WanSchedulerTeaCaching(WanScheduler):
    # 1. 初始化
    def __init__(self, config):
        # 1.1 初始化
        super().__init__(config)

        # 1.2 多一个用于后半段的缓存决策
        self.caching_records_2 = [True] * self.infer_steps

    def clear(self):
        # TODO
        pass

# 1. Taylor: 对单步去噪的前半段，后半段分别缓存
class WanSchedulerTaylorCaching(WanScheduler):
    # 1. 初始化
    def __init__(self, config):
        # 1.1 初始化
        super().__init__(config)

        # 1.2 多一个用于后半段的缓存决策
        pattern = [True, False, False, False]
        self.caching_records = (pattern * ((config.infer_steps + 3) // 4))[:config.infer_steps]
        self.caching_records_2 = (pattern * ((config.infer_steps + 3) // 4))[:config.infer_steps]

    def clear(self):
        # TODO
        pass

# 1. AdaCaching: 对单步去噪的前半段，后半段分别缓存
class WanSchedulerAdaCaching(WanScheduler):
    # 1. 初始化
    def __init__(self, config):
        # 1.1 初始化
        super().__init__(config)

        # 1.2 多一个用于后半段的缓存决策
        self.caching_records_2 = [True] * self.infer_steps

    def clear(self):
        # TODO
        pass