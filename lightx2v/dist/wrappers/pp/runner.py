import gc
import torch
import torch.distributed as dist
from lightx2v.utils.profiler import ProfilingContext4Debug, ProfilingContext
from lightx2v.models.runners.wan.wan_runner import WanRunner
from lightx2v.dist.wrappers.pp.model import PipelineParallelWanModelWrapper
from loguru import logger


class PipelineParallelWanRunnerWrapper:
    def __init__(self, runner: WanRunner, config):
        self.runner = runner
        self.config = config
        
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        self._wrap(self.runner.model, self.config)
        
    def __getattr__(self, name: str):
        if name in self.__dict__:
            return getattr(self, name)
        else:
            return getattr(self.runner, name)
        
    def _wrap(self, model, config):
        model = PipelineParallelWanModelWrapper(model, config)
        
    async def run_pipeline(self):
        if self.config["use_prompt_enhancer"]:
            self.config["prompt_enhanced"] = self.post_prompt_enhancer()
        self.runner.inputs = await self.run_input_encoder()
        kwargs = self.set_target_shape()
        latents, generator = await self.run_dit(kwargs)

        dist.broadcast(latents, src=self.world_size-1)
        
        images = await self.run_vae_decoder(latents, generator)
        self.save_video(images)
        del latents, generator, images
        torch.cuda.empty_cache()
        gc.collect()
        
    def run(self):
        for step_index in range(self.model.scheduler.infer_steps):
            logger.info(f"==> step_index: {step_index + 1} / {self.model.scheduler.infer_steps}")
            
            # for rank_step in range(self.world_size):
            #     if rank_step == 0:
            with ProfilingContext4Debug("step_pre"):
                self.model.scheduler.step_pre(step_index=step_index)

            with ProfilingContext4Debug("infer"):
                self.model.infer(self.inputs)

            # if rank_step == self.world_size - 1:
            with ProfilingContext4Debug("step_post"):
                self.model.scheduler.step_post()

        return self.model.scheduler.latents, self.model.scheduler.generator
            