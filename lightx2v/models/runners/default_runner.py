import asyncio
import gc
import aiohttp
import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from PIL import Image
from lightx2v.utils.profiler import ProfilingContext4Debug, ProfilingContext
from lightx2v.utils.utils import save_videos_grid, cache_video
from lightx2v.utils.generate_task_id import generate_task_id
from lightx2v.utils.envs import *
from lightx2v.utils.service_utils import TensorTransporter, ImageTransporter
from loguru import logger


class DefaultRunner:
    def __init__(self, config):
        self.config = config
        # TODO: implement prompt enhancer
        self.has_prompt_enhancer = False
        # if self.config.prompt_enhancer is not None and self.config.task == "t2v":
        #     self.config["use_prompt_enhancer"] = True
        #     self.has_prompt_enhancer = True
        if self.config["mode"] == "split_server":
            self.model = self.load_transformer()
            self.text_encoders, self.vae_model, self.image_encoder = None, None, None
            self.tensor_transporter = TensorTransporter()
            self.image_transporter = ImageTransporter()
        else:
            self.model, self.text_encoders, self.vae_model, self.image_encoder = self.load_model()

    def set_inputs(self, inputs):
        self.config["prompt"] = inputs.get("prompt", "")
        if self.has_prompt_enhancer and self.config["mode"] != "infer":
            self.config["use_prompt_enhancer"] = inputs.get("use_prompt_enhancer", False)  # Reset use_prompt_enhancer from clinet side.
        self.config["negative_prompt"] = inputs.get("negative_prompt", "")
        self.config["image_path"] = inputs.get("image_path", "")
        self.config["save_video_path"] = inputs.get("save_video_path", "")

    async def post_encoders(self, prompt, img=None, i2v=False):
        tasks = []
        img_byte = self.image_transporter.prepare_image(img) if img is not None else None
        if i2v:
            if "wan2.1" in self.config["model_cls"]:
                tasks.append(
                    asyncio.create_task(
                        self.post_task(task_type="image_encoder", urls=self.config["sub_servers"]["image_encoder"], message={"task_id": generate_task_id(), "img": img_byte}, device="cuda")
                    )
                )
            tasks.append(
                asyncio.create_task(
                    self.post_task(task_type="vae_model/encoder", urls=self.config["sub_servers"]["vae_model"], message={"task_id": generate_task_id(), "img": img_byte}, device="cuda")
                )
            )
        tasks.append(
            asyncio.create_task(
                self.post_task(task_type="text_encoder", urls=self.config["sub_servers"]["text_encoders"], message={"task_id": generate_task_id(), "text": prompt, "img": img_byte}, device="cuda")
            )
        )
        results = await asyncio.gather(*tasks)
        # clip_encoder, vae_encoder, text_encoder
        if not i2v:
            return None, None, results[0]
        if "wan2.1" in self.config["model_cls"]:
            return results[0], results[1], results[2]
        else:
            return None, results[0], results[1]

    async def run_input_encoder(self):
        image_encoder_output = None
        prompt = self.config["prompt_enhanced"] if self.config["use_prompt_enhancer"] else self.config["prompt"]
        i2v = self.config["task"] == "i2v"
        img = Image.open(self.config["image_path"]).convert("RGB") if i2v else None
        with ProfilingContext("Run Encoders"):
            if self.config["mode"] == "split_server":
                clip_encoder_out, vae_encode_out, text_encoder_output = await self.post_encoders(prompt, img, i2v)
                if i2v:
                    if self.config["model_cls"] in ["hunyuan"]:
                        image_encoder_output = {"img": img, "img_latents": vae_encode_out}
                    elif "wan2.1" in self.config["model_cls"]:
                        image_encoder_output = {"clip_encoder_out": clip_encoder_out, "vae_encode_out": vae_encode_out}
                    else:
                        raise ValueError(f"Unsupported model class: {self.config['model_cls']}")
            else:
                if i2v:
                    image_encoder_output = self.run_image_encoder(self.config, self.image_encoder, self.vae_model)
                text_encoder_output = self.run_text_encoder(prompt, self.text_encoders, self.config, image_encoder_output)
        self.set_target_shape()
        self.inputs = {"text_encoder_output": text_encoder_output, "image_encoder_output": image_encoder_output}

        gc.collect()
        torch.cuda.empty_cache()

    def run(self):
        for step_index in range(self.model.scheduler.infer_steps):
            logger.info(f"==> step_index: {step_index + 1} / {self.model.scheduler.infer_steps}")

            with ProfilingContext4Debug("step_pre"):
                self.model.scheduler.step_pre(step_index=step_index)

            with ProfilingContext4Debug("infer"):
                self.model.infer(self.inputs)

            with ProfilingContext4Debug("step_post"):
                self.model.scheduler.step_post()

        return self.model.scheduler.latents, self.model.scheduler.generator

    async def run_step(self, step_index=0):
        self.init_scheduler()
        await self.run_input_encoder()
        self.model.scheduler.prepare(self.inputs["image_encoder_output"])
        self.model.scheduler.step_pre(step_index=step_index)
        self.model.infer(self.inputs)
        self.model.scheduler.step_post()

    def end_run(self):
        self.model.scheduler.clear()
        del self.inputs, self.model.scheduler
        torch.cuda.empty_cache()

    @ProfilingContext("Run VAE")
    async def run_vae(self, latents, generator):
        if self.config["mode"] == "split_server":
            images = await self.post_task(
                task_type="vae_model/decoder",
                urls=self.config["sub_servers"]["vae_model"],
                message={"task_id": generate_task_id(), "latents": self.tensor_transporter.prepare_tensor(latents)},
                device="cpu",
            )
        else:
            images = self.vae_model.decode(latents, generator=generator, config=self.config)
        return images

    @ProfilingContext("Save video")
    def save_video(self, images):
        if not self.config.parallel_attn_type or (self.config.parallel_attn_type and dist.get_rank() == 0):
            if self.config.model_cls in ["wan2.1", "wan2.1_causvid", "wan2.1_skyreels_v2_df"]:
                cache_video(tensor=images, save_file=self.config.save_video_path, fps=self.config.get("fps", 16), nrow=1, normalize=True, value_range=(-1, 1))
            else:
                save_videos_grid(images, self.config.save_video_path, fps=self.config.get("fps", 24))

    async def post_task(self, task_type, urls, message, device="cuda"):
        while True:
            for url in urls:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{url}/v1/local/{task_type}/generate/service_status") as response:
                        status = await response.json()
                    if status["service_status"] == "idle":
                        async with session.post(f"{url}/v1/local/{task_type}/generate", json=message) as response:
                            result = await response.json()
                            if result["kwargs"] is not None:
                                for k, v in result["kwargs"].items():
                                    setattr(self.config, k, v)
                            return self.tensor_transporter.load_tensor(result["output"], device)
            await asyncio.sleep(0.1)

    async def run_pipeline(self):
        if self.config["use_prompt_enhancer"]:
            self.config["prompt_enhanced"] = self.prompt_enhancer(self.config["prompt"])
        self.init_scheduler()
        await self.run_input_encoder()
        self.model.scheduler.prepare(self.inputs["image_encoder_output"])
        latents, generator = self.run()
        self.end_run()
        images = await self.run_vae(latents, generator)
        self.save_video(images)
        del latents, generator, images
        gc.collect()
        torch.cuda.empty_cache()
