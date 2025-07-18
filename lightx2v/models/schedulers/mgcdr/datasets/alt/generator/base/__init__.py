class Generator(object):
    def __init__(self) -> None:
        pass

    def initialize(self, metas, **kwargs):
        raise NotImplementedError

    def generate(self, metas, **kwargs):
        raise NotImplementedError
