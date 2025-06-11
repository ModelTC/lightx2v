from lightx2v.models.schedulers.wan.scheduler import WanScheduler


# 1. TeaCaching: 对单步去噪的前半段，后半段分别缓存
class WanSchedulerTeaCaching(WanScheduler):
    def __init__(self, config):
        super().__init__(config)

    def clear(self):
        # TODO
        pass


# 1. Taylor: 对单步去噪的前半段，后半段分别缓存
class WanSchedulerTaylorCaching(WanScheduler):
    def __init__(self, config):
        super().__init__(config)

        pattern = [True, False, False, False]
        self.caching_records = (pattern * ((config.infer_steps + 3) // 4))[:config.infer_steps]
        self.caching_records_2 = (pattern * ((config.infer_steps + 3) // 4))[:config.infer_steps]

    def clear(self):
        # TODO
        pass
    

# 1. Ada: 对单步去噪的前半段，后半段分别缓存
class WanSchedulerAdaCaching(WanScheduler):
    def __init__(self, config):
        super().__init__(config)

    def clear(self):
        # TODO
        pass


# 1. Custom: 对单步去噪的前半段，后半段分别缓存
class WanSchedulerCustomCaching(WanScheduler):
    def __init__(self, config):
        super().__init__(config)

    def clear(self):
        # TODO
        pass