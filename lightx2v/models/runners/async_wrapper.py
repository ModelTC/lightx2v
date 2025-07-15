import asyncio
from typing import Callable, Any, Optional
from concurrent.futures import ThreadPoolExecutor


class AsyncWrapper:
    def __init__(self, runner, max_workers: Optional[int] = None):
        self.runner = runner
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.executor:
            self.executor.shutdown(wait=True)

    async def run_in_executor(self, func: Callable, *args, **kwargs) -> Any:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args, **kwargs)

    async def run_input_encoder(self):
        if self.runner.config["mode"] == "split_server":
            if self.runner.config["task"] == "i2v":
                return await self.runner._run_input_encoder_server_i2v()
            else:
                return await self.runner._run_input_encoder_server_t2v()
        else:
            if self.runner.config["task"] == "i2v":
                return await self.run_in_executor(self.runner._run_input_encoder_local_i2v)
            else:
                return await self.run_in_executor(self.runner._run_input_encoder_local_t2v)

    async def run_dit(self, kwargs):
        if self.runner.config["mode"] == "split_server":
            return await self.runner._run_dit_server(kwargs)
        else:
            return await self.run_in_executor(self.runner._run_dit_local, kwargs)

    async def run_vae_decoder(self, latents, generator):
        if self.runner.config["mode"] == "split_server":
            return await self.runner._run_vae_decoder_server(latents, generator)
        else:
            return await self.run_in_executor(self.runner._run_vae_decoder_local, latents, generator)

    async def run_prompt_enhancer(self):
        if self.runner.config["use_prompt_enhancer"]:
            return await self.run_in_executor(self.runner.post_prompt_enhancer)
        return None

    async def save_video(self, images):
        return await self.run_in_executor(self.runner.save_video, images)
