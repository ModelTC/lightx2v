from abc import ABC, abstractmethod
import torch
import math

class BaseTransformer(ABC):
    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    @abstractmethod
    def infer(self):
        pass

    @abstractmethod
    def infer_calculating(self):
        pass
        
    @abstractmethod
    def infer_using_cache(self):
        pass

    def taylor_formula(tensor, scheduler) -> torch.Tensor:
        flag = scheduler.step_index
        for i in range(scheduler.step_index, -1, -1):
            if scheduler.caching_records[i]:
                flag = i
        x = scheduler.step_index - flag

        output = 0
        for i in range(len(tensor)):
            output += (1 / math.factorial(i)) * tensor[i] * (x**i)

        return output