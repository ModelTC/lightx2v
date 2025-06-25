from lightx2v.models.runners.wan.wan_runner import WanRunner


class DistriFusionWanRunnerWrapper:
    def __init__(self, runner: WanRunner, config):
        self.runner = runner
        self.config = config
        
    def __getattr__(self, name: str):
        if name in self.__dict__:
            return getattr(self, name)
        else:
            return getattr(self.runner, name)

    def _wrap(self, runner, config):
        runner.model = DistriFusionWanModelWrapper(runner.model, config)
        self.runner.run = self.run