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

    def taylor_formula(self, tensor_dict):
        x = self.get_taylor_step_diff()

        output = 0
        for i in range(len(tensor_dict)):
            output += (1 / math.factorial(i)) * tensor_dict[i] * (x**i)

        return output
    
    def get_taylor_step_diff(self):
        current_step = self.scheduler.step_index
        last_calc_step = current_step - 1
        while last_calc_step >= 0 and not self.scheduler.caching_records[last_calc_step]:
            last_calc_step -= 1
        step_diff = current_step - last_calc_step
        return step_diff